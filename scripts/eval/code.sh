#!/usr/bin/bash

#SBATCH --job-name=moe_eval_code
#SBATCH --output=logs/paper/%x-%j.log
#SBATCH --error=logs/paper/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=auto


nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
num_gpus_per_node=8
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "Node: $head_node"

mbpp() {
    model_id=$1
    model_path=$2

    eval_results_dir="results/${model_id}"
    mkdir -p $eval_results_dir

    torchrun \
    --nnodes 1 \
    --nproc_per_node $num_gpus_per_node \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29522 \
    bigcode-evaluation-harness/main.py \
        --model $model_path \
        --trust_remote_code \
        --tasks mbpp \
        --save_generations \
        --save_references \
        --metric_output_path "$eval_results_dir/mbpp.json" \
        --save_generations_path "$eval_results_dir/generations-mbpp.json" \
        --save_references_path "$eval_results_dir/references-mbpp.json" \
        --temperature 0.1 \
        --do_sample True \
        --n_samples 15 \
        --batch_size 10 \
        --precision bf16 \
        --allow_code_execution
}

humaneval() {
    # generated sheared: 1:07:56
    model_id=$1
    model_path=$2

    eval_results_dir="results/${model_id}"
    mkdir -p $eval_results_dir

    torchrun \
    --nnodes 1 \
    --nproc_per_node $num_gpus_per_node \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29522 \
    bigcode-evaluation-harness/main.py \
        --model $model_path \
        --trust_remote_code \
        --tasks humaneval \
        --save_generations \
        --save_references \
        --metric_output_path "$eval_results_dir/humaneval.json" \
        --save_generations_path "$eval_results_dir/generations-humaneval.json" \
        --save_references_path "$eval_results_dir/references-humaneval.json" \
        --temperature 0.2 \
        --do_sample True \
        --n_samples 200 \
        --batch_size 10 \
        --precision bf16 \
        --allow_code_execution
}

task_name=$1
shift 1

case $task_name in 
    mbpp)
        mbpp $* ;;
    humaneval)
        humaneval $* ;;
    *)
        echo "$task_name not recognized!" ;;
esac
