from dataclasses import dataclass, field

import torch
import gradio as gr
import transformers
from accelerate import Accelerator
from fastchat.train.train import preprocess
from transformers.generation.configuration_utils import GenerationConfig

from src.core.train import get_model_and_tokenizer


tokenizer = None
model = None
accelerator = None


@dataclass
class InferenceArguments:
    model_type: str = field(
        default="moe", metadata={"help": "Model type: `moe` or `auto`"}
    )
    model_name_or_path: str = field(
        default="data/llama_moe", metadata={"help": "Path to model directory"}
    )
    model_max_length: int = field(
        default=1024, metadata={"help": "Maximum length of input sequence"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code during model loading. "
            "This is a security risk, but it is also necessary for loading models from HuggingFace Hub."
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "Padding side: `right` or `left`"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Path to cache directory for model"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype: `float32` or `bfloat16`"},
    )

    def __post_init__(self):
        self.torch_dtype = getattr(torch, self.torch_dtype)


@torch.no_grad()
def chat(message: str, history: list) -> str:
    global tokenizer
    global model
    global accelerator
    if tokenizer is None or model is None or accelerator is None:
        return "Model or Tokenizer or Accelerator not initialized"

    convs = []
    for h in history:
        convs.extend(
            [
                {"from": "human", "value": h[0]},
                {"from": "gpt", "value": h[1]},
            ]
        )
    convs.append({"from": "human", "value": message})
    res = preprocess([convs], tokenizer)
    input_ids = res["input_ids"]
    input_ids = input_ids.to(accelerator.device)
    gen_config = GenerationConfig(
        num_beams=1, do_sample=False, max_new_tokens=tokenizer.model_max_length
    )
    print(f"model device: {model.device}, input_ids device: {input_ids.device}")
    res = model.generate(inputs=input_ids, generation_config=gen_config)
    string = tokenizer.decode(res[0].detach().cpu().tolist())
    return string


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((InferenceArguments,))
    args = parser.parse_args_into_dataclasses()
    infer_args = args[0]
    model, tokenizer = get_model_and_tokenizer(
        infer_args.model_type,
        infer_args.model_name_or_path,
        trust_remote_code=infer_args.trust_remote_code,
        padding_side=infer_args.padding_side,
        model_max_length=infer_args.model_max_length,
        cache_dir=infer_args.cache_dir,
        torch_dtype=infer_args.torch_dtype,
    )
    model.eval()
    accelerator = Accelerator()
    model = accelerator.prepare_model(model)

    gr.ChatInterface(chat).queue().launch(inbrowser=True, share=True)
