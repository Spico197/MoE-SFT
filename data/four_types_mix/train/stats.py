from collections import defaultdict

import matplotlib.pyplot as plt

from src.utils.io import load_jsonlines

if __name__ == "__main__":
    data = load_jsonlines("data/four_types_mix/train/orca/fschat_0.jsonl")
    task_names = set()
    task2num = defaultdict(lambda: 0)

    for ins in data:
        task_names.add(ins["task_name"])
        task2num[ins["task_name"]] += 1
    print(len(data), len(task_names))
    # print(task2num)
    print(sorted(task2num.items(), key=lambda x: x[1]))
    # task_nums = list(task2num.values())
    # plt.hist(task_nums, bins=100)
    # plt.show()
