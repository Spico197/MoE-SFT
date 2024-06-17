#!/usr/bin/bash

#SBATCH --job-name=moe_gen
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --quotatype=auto

mt_bench() {
    model_id=$1
    model_path=$2
    shift 2

    python -m src.eval.gen_mt_ans \
        --model-id $model_id \
        --model-path $model_path $*
}

alpaca_eval() {
    model_id=$1
    model_path=$2
    shift 2

    python -m src.eval.gen_alpaca_eval_ans \
        --model-id $model_id \
        --model-path $model_path $*
}

task_name=$1
shift 1

case $task_name in 
    mt_bench)
        mt_bench $* ;;
    alpaca_eval)
        alpaca_eval $* ;;
    *)
        echo "$task_name not recognized!" ;;
esac
