import pathlib

from src.utils.io import load_jsonlines, dump_jsonlines


def split_data(input_filepath, output_dir, num_splits: int):
    data = load_jsonlines(input_filepath)
    num_ins = len(data)
    num_ins_per_split = num_ins // num_splits
    for i in range(num_splits):
        split_output_dir = pathlib.Path(output_dir).joinpath(f"split_{i}")
        split_output_dir.mkdir(parents=True, exist_ok=True)
        start = i * num_ins_per_split
        end = (i + 1) * num_ins_per_split if i != num_splits - 1 else num_ins
        split_data = data[start:end]
        dump_jsonlines(split_data, f"{str(split_output_dir)}/fschat_0.jsonl")


if __name__ == "__main__":
    split_data("data/merged/all/fschat_0.jsonl", "data/merged_splits", 8)
