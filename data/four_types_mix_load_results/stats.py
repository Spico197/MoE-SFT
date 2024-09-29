import json
import pathlib
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.io import load_jsonlines


def cv_squared(gate_load: list, eps=1e-10):
    load = torch.tensor(gate_load, dtype=torch.float)
    if load.shape[0] == 1:
        return 0.0
    val = load.var() / (load.mean() ** 2 + eps)
    return val


def aggregate_data(
    raw_data_filepath: str,
    input_folder: str,
    output_folder: str,
    cv_squared_range: tuple = (0, 0.2, 0.025),
):
    """Aggregating data with similar gate loads into the same set."""
    input_folder_p = pathlib.Path(input_folder)
    output_folder_p = pathlib.Path(output_folder)
    output_folder_p.mkdir(parents=True, exist_ok=True)

    id2raw_data = {}
    for ins in load_jsonlines(raw_data_filepath):
        id2raw_data[ins["id"]] = ins
    cv_range_to_data = defaultdict(list)
    interval = cv_squared_range[2]
    p = cv_squared_range[0]
    while p < cv_squared_range[1]:
        # (0, 0.025]
        cv_range_to_data[(p, p + interval)] = []
        p += interval

    jsonl_files = list(input_folder_p.glob("split_*.jsonl"))
    for jsonl_file in tqdm(jsonl_files, desc="Reading"):
        data_split = load_jsonlines(jsonl_file)
        for ins in data_split:
            cv_val = cv_squared(ins["gate_load"])
            raw_ins = id2raw_data[ins["id"]]
            for k in cv_range_to_data:
                if k[0] < cv_val <= k[1]:
                    cv_range_to_data[k].append(raw_ins)
                    break

    for k in tqdm(cv_range_to_data, desc="Dumping"):
        out_file = output_folder_p.joinpath(f"{round(k[0], 5)}_{round(k[1], 5)}.jsonl")
        with out_file.open("w", encoding="utf8") as f:
            for ins in cv_range_to_data[k]:
                f.write(f"{json.dumps(ins, ensure_ascii=False)}\n")


def main():
    stats = load_jsonlines(
        "data/merged_splits_gate_load_results/split_3_gate_load.jsonl"
    )
    gate_load_list = [s["gate_load"] for s in stats]
    cv_list = [cv_squared(gate_load) for gate_load in gate_load_list]

    plt.hist(cv_list, bins=100)
    plt.xlim(0, 0.2)
    plt.savefig("data/merged_splits_gate_load_results/split_3_cv_hist.png")


if __name__ == "__main__":
    # main()
    aggregate_data(
        "data/merged/all/fschat_0.jsonl",
        "data/merged_splits_gate_load_results/",
        "data/merged_splits_gate_load_data_aggregated/",
        (0, 0.1, 0.0125)
    )
