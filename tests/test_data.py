import transformers

from src.data import (
    SubDirWeightedPackedJsonlDataset,
    get_uniform_sampling_ratio,
    fault_tolerance_data_collator,
    preprocess,
)


def test_data_sampling():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/mnt/petrelfs/zhutong/llama-moe-models/LLaMA-MoE-v1-3_5B-2_8",
        model_max_length=64,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SubDirWeightedPackedJsonlDataset(
        "data/merged",
        tokenizer,
        prob_map=get_uniform_sampling_ratio("data/merged"),
        seed=1227,
    )

    i = 5
    for ins in train_dataset:
        if i <= 0:
            break
        print(ins)
        # i -= 1


def test_preprocess():
    tokenizer_dir = "/mnt/petrelfs/zhutong/llama-moe-models/LLaMA-MoE-v1-3_5B-2_8-new"
    # tokenizer_dir = "/mnt/petrelfs/share_data/quxiaoye/models/MoLM-700M-4B"

    # instances = load_jsonlines(datapath)
    ins = {
        "conversations": [
            {"from": "human", "value": "Using the Pandas library, write a code to read a table from a web page and print out some of its rows.\nurl = 'https://en.wikipedia.org/wiki/List_of_U.S._states_by_GDP_per_capita'"},
            {"from": "gpt", "value": "import pandas as pd\n\n# Read table from the given url\ndf = pd.read_html(url)[1]\n\n# Print out some of the rows\nprint(df.head())"},
        ]
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir,
        padding_side="right",
        model_max_length=2048,
        use_fast=False,
        # legacy=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    ins["conversations"] += (ins["conversations"] + ins["conversations"] + ins["conversations"])
    res = preprocess([ins], tokenizer)
    table = []
    for i in range(len(res["input_ids"][0])):
        row = (i, tokenizer.convert_ids_to_tokens([res["input_ids"][0][i]])[0], res["labels"][0][i].item(), res["attention_mask"][0][i].item())
        table.append(row)
    print(table)


if __name__ == "__main__":
    test_preprocess()
