import torch

from src.models import ModuleFormerConfig, ModuleFormerForCausalLM


def test_moduleformer():
    model_dir = "/mnt/petrelfs/share_data/quxiaoye/models/MoLM-700M-4B"
    config = ModuleFormerConfig.from_pretrained(model_dir)
    config.n_layer = 2
    m = ModuleFormerForCausalLM.from_pretrained(model_dir, config=config)
    print(m)

    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 5:] = 0
    outs = m(input_ids, attention_mask=attention_mask)


if __name__ == "__main__":
    test_moduleformer()
