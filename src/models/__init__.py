from .llama_moe import LlamaMoEConfig, LlamaMoEForCausalLM
from .moduleformer import ModuleFormerConfig, ModuleFormerForCausalLM


MODEL_CONFIG_MAP = {
    "llama_moe": (LlamaMoEConfig, LlamaMoEForCausalLM),
    "moduleformer": (ModuleFormerConfig, ModuleFormerForCausalLM),
}
