import json
import math
import pathlib

import torch
import transformers
from torch.utils.data import DataLoader
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model

# from src.utils.io import dump_jsonlines
from src.utils.config import ModelArguments, DataArguments, TrainingArguments
from src.data import (
    fault_tolerance_data_collator_with_str_fields,
    CachedJsonlWithDataIdDataset,
)
from src.models import MODEL_CONFIG_MAP


def get_tokenizer(
    model_name_or_path,
    cache_dir: str = None,
    model_max_length: int = 2048,
    padding_side: str = "right",
    use_fast: bool = False,
    trust_remote_code: bool = False,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer ready, pad_token: {tokenizer.pad_token}")
    return tokenizer


def get_model(
    model_type: str,
    model_name_or_path: str,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    additional_config: dict = None,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 32.0,
    lora_modules_to_save: str = "embed_tokens,lm_head",
    lora_trainable: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
    lora_dropout: float = 0.1,
    bnb_bits: int = 16,
    bnb_double_quant: bool = True,
    bnb_quant_type: str = "nf4",
):
    logger.info(f"Model type: {model_type}")
    if model_type == "auto":
        ConfigClass = transformers.AutoConfig
        ModelClass = transformers.AutoModelForCausalLM
    elif model_type in MODEL_CONFIG_MAP:
        ConfigClass, ModelClass = MODEL_CONFIG_MAP[model_type]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set RoPE scaling factor
    config = ConfigClass.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    if additional_config is not None:
        config.update(additional_config)
    logger.info("Config ready")

    # Load model and tokenizer
    model = ModelClass.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        load_in_4bit=bnb_bits == 4,
        load_in_8bit=bnb_bits == 8,
        quantization_config=(
            transformers.BitsAndBytesConfig(
                load_in_4bit=bnb_bits == 4,
                load_in_8bit=bnb_bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=bnb_double_quant,
                bnb_4bit_quant_type=bnb_quant_type,
            )
            if use_lora
            else None
        ),
    )
    if use_lora:
        logger.info(
            f"LoRA model: r={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout},"
            f" modules_to_save={lora_modules_to_save}, trainable={lora_trainable}"
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_trainable.split(","),
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=lora_modules_to_save,
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model = get_peft_model(model, peft_config)
    logger.info("model ready")

    return model


def get_model_and_tokenizer(
    model_type: str,
    model_name_or_path: str,
    tokenizer_path: str = None,
    torch_dtype: str = "auto",
    model_max_length: int = 2048,
    attn_impl: str = "flash_attention_2",
    cache_dir: str = None,
    trust_remote_code: bool = False,
    padding_side: str = "right",
    additional_config: dict = None,
    use_fast: bool = False,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 32.0,
    lora_modules_to_save: str = "embed_tokens,lm_head",
    lora_trainable: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
    lora_dropout: float = 0.1,
    bnb_bits: int = 16,
    bnb_double_quant: bool = True,
    bnb_quant_type: str = "nf4",
) -> tuple:
    if tokenizer_path is None:
        tokenizer_path = model_name_or_path
    tokenizer = get_tokenizer(
        tokenizer_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    model = get_model(
        model_type,
        model_name_or_path,
        torch_dtype=torch_dtype,
        model_max_length=model_max_length,
        attn_impl=attn_impl,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        additional_config=additional_config,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_modules_to_save=lora_modules_to_save,
        lora_trainable=lora_trainable,
        lora_dropout=lora_dropout,
        bnb_bits=bnb_bits,
        bnb_double_quant=bnb_double_quant,
        bnb_quant_type=bnb_quant_type,
    )

    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    training_args.report_to = list(set(training_args.report_to) | {"tensorboard"})
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    ac = Accelerator()

    model, tokenizer = get_model_and_tokenizer(
        model_args.model_type,
        model_args.model_name_or_path,
        tokenizer_path=model_args.tokenizer_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=model_args.padding_side,
        torch_dtype=model_args.torch_dtype,
        additional_config=model_args.additional_config,
        attn_impl=model_args.attn_impl,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        use_lora=model_args.use_lora,
        lora_rank=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_modules_to_save=model_args.lora_modules_to_save,
        lora_trainable=model_args.lora_trainable,
        lora_dropout=model_args.lora_dropout,
        bnb_bits=model_args.bnb_bits,
        bnb_double_quant=model_args.bnb_double_quant,
        bnb_quant_type=model_args.bnb_quant_type,
    )
    logger.info(
        f"tokenizer.pad_token = {tokenizer.pad_token}, token id = {tokenizer.pad_token_id}"
    )
    model.eval()
    model = ac.prepare_model(model)
    eval_dataset = CachedJsonlWithDataIdDataset(
        data_args.dataset_dir_or_path,
        tokenizer,
        seed=training_args.seed,
    )
    data_type = pathlib.Path(data_args.dataset_dir_or_path).parent.name
    eval_data_dir_p = pathlib.Path(data_args.eval_data_dir)
    eval_data_dir_p.mkdir(parents=True, exist_ok=True)
    result_path = eval_data_dir_p / f"{data_type}_gate_load.jsonl"
    loader = DataLoader(
        eval_dataset,
        batch_size=1,  # must be 1
        collate_fn=fault_tolerance_data_collator_with_str_fields,
    )
    eval_dataloader = ac.prepare_data_loader(loader)
    logger.info("data ready")
    num_ins = 0
    with torch.inference_mode():
        with result_path.open("w", encoding="utf8") as f:
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                outs = model(**batch, output_attentions=False, use_cache=False)
                gate_load = outs.gate_load
                gate_load = gate_load[-1].detach().cpu().numpy().tolist()
                ins = {"id": batch["id"][0], "gate_load": gate_load}
                num_ins += 1
                f.write(f"{json.dumps(ins, ensure_ascii=False)}\n")
                f.flush()
        logger.info(f"{num_ins} results saved to {result_path}")

    logger.info("🎉 All done~")


if __name__ == "__main__":
    train()
