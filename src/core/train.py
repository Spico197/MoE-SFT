import math
import pathlib

import torch
import transformers
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model

from src.utils.config import ModelArguments, DataArguments, TrainingArguments
from src.data import (
    SubDirWeightedPackedJsonlDataset,
    fault_tolerance_data_collator,
    CachedJsonlDataset,
    get_cached_datasets_from_dir,
)
from src.utils.io import trainer_save_model_safe
from src.models import MODEL_CONFIG_MAP
from src.trainer import GateLoadRecordingTrainer
from src.callbacks import (
    AdaptiveSamplingCallback,
    RefLossSamplingCallback,
    RandomSamplingCallback,
    RandomBatchSamplingCallback,
    CycleSamplingCallback,
    SavePeftModelCallback,
)


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
    gate_exclude_list = ["gate_proj", "weight_gate"]
    if training_args.freeze_gate:
        for name, param in model.named_parameters():
            # if "gate" in name and all(n not in name for n in gate_exclude_list):
            if "gate" in name:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(f"{name} - Grad: {param.requires_grad} ({param.numel()})")

    train_dataset = None
    datapath = pathlib.Path(data_args.dataset_dir_or_path)
    if not datapath.exists():
        raise ValueError(f"Dataset path {datapath} not found")
    elif datapath.is_dir():
        logger.info(f"SubDirWeightedPackedJsonlDataset: {datapath}")
        train_dataset = SubDirWeightedPackedJsonlDataset(
            data_args.dataset_dir_or_path,
            tokenizer,
            # prob_map=get_uniform_sampling_ratio(data_args.dataset_dir_or_path),
            # prob_map={"code": 0.25119094959816823, "math": 0.2674581878910902, "orca": 0.243050776175138, "sharegpt": 0.23830008633560357},
            prob_map=data_args.prob_map,
            seed=training_args.seed,
        )
    elif datapath.is_file():
        logger.info(f"CachedJsonlDataset: {datapath}")
        train_dataset = CachedJsonlDataset(
            data_args.dataset_dir_or_path,
            tokenizer,
            seed=training_args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset path type: {datapath}")
    logger.info("train dataset ready")

    eval_dataset = None
    if data_args.eval_data_dir is not None and (
        training_args.do_eval or training_args.do_final_eval
    ):
        eval_dataset = get_cached_datasets_from_dir(
            data_args.eval_data_dir, tokenizer, seed=training_args.seed
        )
        eval_datanames = sorted(eval_dataset.keys())
        train_datanames = sorted(train_dataset.data_type_to_dataset.keys())
        assert eval_datanames == train_datanames, (
            f"Eval dataset names {eval_datanames} "
            f"do not match train dataset names {train_datanames}"
        )
        logger.info("eval dataset ready (for dynamic sampling usage)")

    trainer = GateLoadRecordingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=fault_tolerance_data_collator,
    )
    if model_args.use_lora:
        trainer.add_callback(SavePeftModelCallback)

    if training_args.do_eval:
        if training_args.dynamic_sampling_type == "load":
            callback = AdaptiveSamplingCallback(
                criterion=training_args.dynamic_sampling_criterion,
                sim_type=training_args.dynamic_sampling_sim_type,
                eta=training_args.dynamic_eta,
                c=training_args.dynamic_c,
            )
        elif training_args.dynamic_sampling_type == "loss":
            callback = RefLossSamplingCallback(
                training_args.dynamic_sampling_name2ref_loss,
                eta=training_args.dynamic_eta,
                c=training_args.dynamic_c,
            )
        elif training_args.dynamic_sampling_type == "random":
            callback = RandomSamplingCallback()
        elif training_args.dynamic_sampling_type == "random_batch":
            callback = RandomBatchSamplingCallback()
        elif training_args.dynamic_sampling_type == "cycle":
            callback = CycleSamplingCallback()
        else:
            raise ValueError(
                f"Unknown dynamic sampling type: {training_args.dynamic_sampling_type}"
            )
        trainer.add_callback(callback)
    logger.info("trainer ready")

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logger.info("resume training from ckpt")
            trainer.train(resume_from_checkpoint=True)
        else:
            logger.info("start training")
            trainer.train()

    # Save model
    if training_args.save_final_ckpt:
        logger.info("training finished, dumping model")
        model.config.use_cache = True
        trainer.save_state()
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)

    if training_args.do_final_eval:
        metrics = None
        if isinstance(trainer.eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
                dataset_metrics = trainer.evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=None,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(ignore_keys=None)
        metrics.update({"all_metrics": True})
        logger.info(f"Final eval metrics: {metrics}")

    logger.info("🎉 All done~")


if __name__ == "__main__":
    train()
