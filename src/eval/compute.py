import argparse
import statistics
from pathlib import Path

from src.utils.io import load_json, dump_json
from src.eval.show import TASK_MAP


def exact_match_strip_resp(task: str, result_dir: str):
    result_dir_p = Path(result_dir)
    results = {}
    tot_vals = []
    for alias in TASK_MAP[task].split(","):
        p = list(result_dir_p.glob(f"*{alias}.jsonl"))
        if len(p) == 1:
            p = p[0]
            res = load_json(p)
            tot = correct = 0
            for doc in res:
                tot += 1
                if (
                    len(doc["filtered_resps"]) == 1
                    and doc["filtered_resps"][0].strip() == doc["doc"]["target"]
                ):
                    correct += 1
            if tot > 0:
                val = correct / tot
            else:
                val = 0.0
            tot_vals.append(val)
            results[alias] = {
                "exact_match,none": val,
                "alias": alias,
            }
    final_results = {"results": results}
    print(
        f"{result_dir} - task: {task}, num: {len(TASK_MAP[task].split(','))}, avg: {100 * statistics.mean(tot_vals):.3f} %"
    )
    dump_json(final_results, result_dir_p / f"{task}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="task name")
    parser.add_argument("result_dir", type=str, help="result directory")
    args = parser.parse_args()

    exact_match_strip_resp(args.task, args.result_dir)
