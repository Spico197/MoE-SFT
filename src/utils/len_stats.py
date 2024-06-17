import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from fastchat.conversation import get_conv_template

from src.utils.io import load_jsonlines


tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/petrelfs/zhutong/llama-moe-models/LLaMA-MoE-v1-3_5B-2_8"
)


def len_stat(filepath):
    data = load_jsonlines(filepath)
    len_list = []
    for d in data:
        for turn in d["choices"][0]["turns"]:
            _len = len(tokenizer.encode(turn))
            len_list.append(_len)
    print(filepath)
    print("max: ", max(len_list))
    print("min: ", min(len_list))
    print("mean: ", sum(len_list) / len(len_list))
    return len_list


def fschat_len_stat(filepath, template: str = "vicuna_v1.1"):
    data = load_jsonlines(filepath)
    t = get_conv_template(template)
    len_list = []
    for ins in tqdm(data):
        for turn in ins["conversations"]:
            t.append_message(turn["from"], turn["value"])
        conv_string = t.get_prompt()
        t.messages.clear()
        _len = len(tokenizer.encode(conv_string))
        len_list.append(_len)
    print(filepath)
    print("max: ", max(len_list))
    print("min: ", min(len_list))
    print("mean: ", sum(len_list) / len(len_list))
    return len_list


if __name__ == "__main__":
    # lens_llama_moe = len_stat("data/mt_bench/model_answer/llama_moe_sharegpt.jsonl")
    # lens_sheared_llama = len_stat("data/mt_bench/model_answer/sheared_llama_sharegpt.jsonl")

    lens_fschat = fschat_len_stat("data/open_orca/merged_samples_-1/fschat_0.jsonl")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lens, label in [
        # (lens_llama_moe, "llama_moe"),
        # (lens_sheared_llama, "sheared_llama"),
        (lens_fschat, "open_orca/merged_samples_-1"),
    ]:
        ax.hist(lens, bins=100, alpha=0.5, label=label)
    ax.legend()
    ax.set_xlabel("length")
    ax.set_ylabel("count")
    # plt.savefig("data/mt_bench/model_answer/len_stats.png")
    plt.savefig("data/open_orca/merged_samples_-1/len_stats.png")
