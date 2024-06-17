"""
nohup python listen_eval.py > eval_listen.log 2>&1 &
"""

import os
import re
import time
import argparse
import subprocess
from pathlib import Path
from typing import List

from loguru import logger

from src.utils.notification import send_to_wechat

logger.add("listen.log")


def run_command(command):
    try:
        logger.info(f"Running cmd: {command}")
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")


def eval_one(
    abbr: str,
    folder: str,
    partition: str = "llm_x",
    tasks: List[str] = ["qa", "bbh", "reasoning", "mmlu", "mbpp", "humaneval"],
):
    for task in tasks:
        logger.info(f"Evaluating task {task} for {abbr}, folder: {str(folder)}")

        if task in ["mbpp", "humaneval"]:
            run_command(f"sbatch scripts/eval/code.sh {task} {abbr} {folder}")
        else:
            run_command(
                f"nohup srun -p {partition} --gres=gpu:1 bash scripts/eval/eval.sh {task} {folder} True results/{abbr} 1>logs/paper/eval-{abbr}-{task}.log 2>&1 &"
            )


def get_job_status(jobid: str):
    cmd = f"sacct -j {jobid} -o state -n"
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    ret = p.stdout.read().decode("utf8").strip()
    return ret


def submit_job(script: str):
    cmd = f"sbatch {script}"
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    ret = p.stdout.read().decode("utf8").strip()
    # extract job id from `Submitted batch job 2524663` with regex
    matched = re.search(r"\d+", ret)
    if matched:
        jobid = matched.group()
    else:
        jobid = None
    return jobid


def job_is_runnnig(jobid: str):
    status = get_job_status(jobid)
    return "RUN" in status or "PEND" in status


def has_ckpt(folder: Path):
    if isinstance(folder, str):
        folder = Path(folder)
    bin_models = list(folder.glob("pytorch_model*.bin"))
    sf_models = list(folder.glob("model*.safetensors"))
    return len(bin_models) > 0 or len(sf_models) > 0


def make_notice(notice: str):
    send_to_wechat(notice)
    logger.info(notice)


def listen(
    abbr: str,
    folder: str,
    jobid: str = None,
    partition: str = "MoE",
    tasks: List[str] = ["qa", "bbh", "reasoning", "mmlu", "mbpp", "humaneval"],
    skip_wait: bool = False,
    auto_restart: bool = False,
):
    """
    Args:
        evaluated: list of (ckpt_id, task) representing the evaluated tasks.
    """
    sleep_interval = 10 * 60
    results_folder = Path(f"results/{abbr}")
    log_dir = "logs/paper"

    notice = f"Listening model results for {abbr} {jobid} ({tasks}) in {folder} - pid: {os.getpid()}"
    make_notice(notice)

    Path(log_dir).mkdir(exist_ok=True, parents=True)
    results_folder.mkdir(exist_ok=True, parents=True)
    folder = Path(folder)
    while True:
        if jobid:
            if not job_is_runnnig(jobid):
                notice = ""
                err = False
                if has_ckpt(folder):
                    notice = f"FINISHED: Job {abbr}({jobid}) finished"
                else:
                    err = True
                    notice = f"ERR: Job {abbr}({jobid}) exit without ckpt"
                make_notice(notice)
                if err:
                    if auto_restart:
                        sbatch_p = folder.joinpath("sbatch.sh")
                        if sbatch_p.exists():
                            new_jobid = submit_job(str(sbatch_p))
                            if new_jobid is None:
                                make_notice(
                                    f"FAILED RESTART: job {abbr} {jobid} in {folder}"
                                )
                            else:
                                new_folder = str(folder).replace(jobid, new_jobid)
                                make_notice(
                                    f"RESTART: job {abbr} {jobid} in {folder} -> {new_jobid} in {new_folder}"
                                )
                                folder = Path(new_folder)
                                jobid = new_jobid
                        else:
                            make_notice(
                                f"FAILED RESTART: job {abbr} {jobid} in {folder}, sbatch.sh not found!"
                            )
                    else:
                        make_notice(
                            f"FAILED w/o RESTART: job {abbr} {jobid} in {folder} failed, please check it manually!"
                        )
                        break
        if has_ckpt(folder):
            make_notice(f"DETECTED: Ckpt detected in {str(folder)}")
            if not skip_wait:
                logger.info(
                    f"Sleep for {sleep_interval} seconds to avoid incomplete dumping"
                )
                time.sleep(sleep_interval)
            eval_one(abbr, folder, partition=partition, tasks=tasks)
            break
        time.sleep(sleep_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("abbr", type=str)
    parser.add_argument("folder", type=str)
    parser.add_argument(
        "-p", "--partition", type=str, default="MoE", help="slurm partition"
    )
    parser.add_argument(
        "-j", "--jobid", type=str, default=None, help="slurm job id to watch"
    )
    parser.add_argument("--skip_wait", action="store_true", default=False)
    parser.add_argument("--auto_restart", action="store_true", default=False)
    parser.add_argument("--tasks", type=str, default="qa,bbh,reasoning,mmlu,mbpp")
    args = parser.parse_args()

    tasks = args.tasks.split(",")

    listen(
        args.abbr,
        args.folder,
        jobid=args.jobid,
        tasks=tasks,
        partition=args.partition,
        skip_wait=args.skip_wait,
        auto_restart=args.auto_restart,
    )
