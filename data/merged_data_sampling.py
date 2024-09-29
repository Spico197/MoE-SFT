from pathlib import Path
from collections import defaultdict

from loguru import logger

from src.utils.io import load_jsonlines, dump_jsonlines


def sample(filepath: str, task_num: int, output_dir: str):
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(exist_ok=True, parents=True)

    data = load_jsonlines(filepath)
    task2num = defaultdict(lambda: 0)
    task2data = defaultdict(list)
    for ins in data:
        task = ins["task_name"].replace("/", "_").replace("-", "_").replace(":", "_").replace(".", "_")
        task2num[task] += 1
        task2data[task].append(ins)
    logger.info(f"Task num: {len(task2num)}, total ins num: {len(data)}")
    topk_tasks = sorted(task2num.items(), key=lambda x: x[1], reverse=True)[:task_num]
    logger.info(f"topk_tasks: {topk_tasks}")
    for task, _ in topk_tasks:
        num_ins = len(task2data[task])
        train_num = int(num_ins * 0.8)
        task_data_train = task2data[task][:train_num]
        task_data_dev = task2data[task][train_num:]
        output_dir_p.joinpath("train").joinpath(task).mkdir(exist_ok=True, parents=True)
        output_dir_p.joinpath("dev").mkdir(exist_ok=True, parents=True)
        dump_jsonlines(
            task_data_train,
            output_dir_p.joinpath("train").joinpath(task).joinpath("fschat_0.jsonl"),
        )
        dump_jsonlines(
            task_data_dev,
            output_dir_p.joinpath("dev").joinpath(f"{task}.jsonl"),
        )


if __name__ == "__main__":
    # for task_num in [50, 100, 200, 300, 400, 500]:
    for task_num in [5, 10, 15, 20, 25, 30]:
        sample(
            "data/merged/all/fschat_0.jsonl",
            task_num,
            f"data/merged_sampling/task_{task_num}",
        )
