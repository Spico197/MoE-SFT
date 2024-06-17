import pathlib
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import transformers

from src.utils.io import load_json
from src.data import get_uniform_sampling_ratio


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    model_type: str = field(
        default="moe", metadata={"help": "Model type: `moe` or `auto`"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "Torch dtype: `float32` or `bfloat16`"},
    )
    additional_config: str = field(
        default=None,
        metadata={"help": "Additional config file (in json) to load"},
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={
            "help": "attention implementation, choice from [eager, flash_attention_2, sdpa, none] (default: `flash_attention_2`)"
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA"},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "rank of LoRA"},
    )
    lora_alpha: float = field(
        default=32.0,
        metadata={"help": "alpha of LoRA"},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout of LoRA"},
    )
    lora_modules_to_save: str = field(
        default=None,
        metadata={"help": "Modules to save in the final checkpoint (for LoRA usage)"},
    )
    lora_trainable: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        metadata={"help": "Trainable modules (for LoRA usage)"},
    )
    bnb_double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    bnb_quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bnb_bits: int = field(default=16, metadata={"help": "How many bits to use."})

    def __post_init__(self):
        if hasattr(torch, self.torch_dtype):
            self.torch_dtype = getattr(torch, self.torch_dtype)
        if self.additional_config is not None:
            if not pathlib.Path(self.additional_config).exists():
                raise ValueError(
                    f"Additional config file {self.additional_config} not found"
                )
            self.additional_config = load_json(self.additional_config)
        if self.attn_impl.lower() in ["none", "null", "no", "n/a"]:
            self.attn_impl = None


@dataclass
class DataArguments:
    eval_data_dir: str = field(
        default=None, metadata={"help": "Path to the evaluation data folder."}
    )
    dataset_dir_or_path: str = field(
        default="data/merged",
        metadata={"help": "Path to dataset directory or a single jsonl file"},
    )
    prob_map: str = field(
        default=None,
        metadata={"help": "Path to the probability map file"},
    )

    def __post_init__(self):
        if self.prob_map is not None:
            if not pathlib.Path(self.prob_map).exists():
                raise ValueError(f"Probability map file {self.prob_map} not found")
            self.prob_map = load_json(self.prob_map)
        elif pathlib.Path(self.dataset_dir_or_path).is_dir():
            self.prob_map = get_uniform_sampling_ratio(self.dataset_dir_or_path)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_eval_steps_per_type: int = field(
        default=10,
        metadata={"help": "Maximum number of steps to perform during evaluation."},
    )
    dynamic_sampling_type: Literal[
        "load", "loss", "random", "random_batch", "cycle"
    ] = field(
        default="load",
        metadata={
            "help": "The type of dynamic sampling. Choices: `load`, `loss`, `random`, `random_batch`, `cycle`"
        },
    )
    dynamic_sampling_name2ref_loss: str = field(
        default=None,
        metadata={
            "help": "The filepath to reference loss mapping for dynamic sampling (loss)."
        },
    )
    dynamic_sampling_sim_type: Literal["cos", "l2", "~l2"] = field(
        default="l2",
        metadata={
            "help": "The similarity metric to use for dynamic sampling. Choices: `cos`, `l2`"
        },
    )
    dynamic_sampling_criterion: Literal["min", "max", "mean"] = field(
        default="mean",
        metadata={
            "help": "The criterion of sim/delta to use for dynamic sampling. Choices: `min`, `max`, `mean`"
        },
    )
    dynamic_eta: float = field(
        default=10.0,
        metadata={"help": "change rate of weights updating"},
    )
    dynamic_c: float = field(
        default=5e-2,
        metadata={"help": "smoothing value"},
    )
    freeze_gate: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the gate during training."},
    )
    do_final_eval: bool = field(
        default=True,
        metadata={"help": "Whether to perform final evaluation."},
    )
    save_final_ckpt: bool = field(
        default=True,
        metadata={"help": "Whether to save final checkpoint."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.dynamic_sampling_name2ref_loss is not None:
            if not pathlib.Path(self.dynamic_sampling_name2ref_loss).exists():
                raise ValueError(
                    f"Probability map file {self.dynamic_sampling_name2ref_loss} not found"
                )
            self.dynamic_sampling_name2ref_loss = load_json(
                self.dynamic_sampling_name2ref_loss
            )
