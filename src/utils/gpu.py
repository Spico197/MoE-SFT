import time
import argparse

from loguru import logger
from gpustat.core import GPUStatCollection


def check_gpu(devices: str) -> bool:
    """
    Check if the GPU is available and the devices are empty without processes.
    This is determined by the physical GPU devices and not not the CUDA_VISIBLE_DEVICES.
    """
    gs = GPUStatCollection.new_query()
    gpu_indices = [int(i) for i in devices.split(",")]
    process_nums = []
    for i in gpu_indices:
        gpu = gs.gpus[i]
        process_nums.append(len(gpu["processes"]))
    if all(x == 0 for x in process_nums):
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "devices",
        type=str,
        help="The GPU devices to use.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="The interval to check the GPU devices.",
    )
    args = parser.parse_args()

    logger.info("Checking the GPU devices", end="")
    while not check_gpu(args.devices):
        print(".", end="")
        time.sleep(args.interval)
    logger.info(f"GPU devices {args.devices} are available now.")
