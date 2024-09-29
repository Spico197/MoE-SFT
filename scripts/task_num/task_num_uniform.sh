#!/usr/bin/bash

#SBATCH --job-name=llama_moe_uniform_task_num
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --nodes=1
#SBATCH -w SH-IDCA1404-10-140-54-12
##SBATCH --gres=gpu:4
##SBATCH --quotatype=auto

set -x

export WANDB_PROJECT="adaptive-moe-sft"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=4


note() {
    msg=$1
    echo $msg
    python -m src.utils.notification feishu "$msg"
}

mbpp() {
    model_id=$1
    model_path=$2

    note "Start MBPP evaluation for $model_id on $model_path, wait for devices"
    python -m src.utils.gpu $CUDA_VISIBLE_DEVICES
    note "MBPP device OK - $CUDA_VISIBLE_DEVICES"

    eval_results_dir="results/${model_id}"
    mkdir -p $eval_results_dir

    torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
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

multi_eval() {
    # task_name is the model_id in mbpp
    task_name=$1
    model_path=$2
    mbpp $task_name $model_path
    sleep 1
    note "Start qa evaluation for $model_id on $model_path, wait for devices"
    python -m src.utils.gpu "0"
    note "qa device OK - 0"
    CUDA_VISIBLE_DEVICES=0 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh qa $model_path True results/$task_name 1>logs/eval-$task_name-qa.log 2>&1 &
    sleep 1
    note "Start bbh evaluation for $model_id on $model_path, wait for devices"
    python -m src.utils.gpu "1"
    note "bbh device OK - 1"
    CUDA_VISIBLE_DEVICES=1 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh bbh $model_path True results/$task_name 1>logs/eval-$task_name-bbh.log 2>&1 &
    sleep 1
    note "Start reasoning evaluation for $model_id on $model_path, wait for devices"
    python -m src.utils.gpu "2"
    note "reasoning device OK - 2"
    CUDA_VISIBLE_DEVICES=2 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh reasoning $model_path True results/$task_name 1>logs/eval-$task_name-reasoning.log 2>&1 &
    sleep 1
    note "Start mmlu evaluation for $model_id on $model_path, wait for devices"
    python -m src.utils.gpu "3"
    note "mmlu device OK - 3"
    CUDA_VISIBLE_DEVICES=3 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh mmlu $model_path True results/$task_name 1>logs/eval-$task_name-mmlu.log 2>&1 &
}

task_num_job() {
    num_tasks=$1
    task_name="uniform_task_$num_tasks"
    model_type="auto"
    model_name_or_path="data/llama-moe-models/LLaMA-MoE-v1-3_5B-2_8-new"
    dataset_dir_or_path="data/merged_sampling/task_$num_tasks/train"
    eval_data_dir="data/merged_sampling/task_$num_tasks/dev"

    note "Start training job ${task_name}, wait for devices"
    python -m src.utils.gpu $CUDA_VISIBLE_DEVICES
    note "training job ${task_name} device OK - ${CUDA_VISIBLE_DEVICES}"

    comment="llama-moe 2/8, four type mix, uniformly sampling, 4 gpus, eval_steps 100, max_eval_steps 5, w/ balance loss, w/ freeze gate, w/ gate noise"
    base_dir="outputs"
    output_dir="${base_dir}/${task_name}/$SLURM_JOB_ID"
    mkdir -p $output_dir
    scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    git diff > $output_dir/diff.patch
    env > $output_dir/env
    echo -e "Job ID: ${SLURM_JOB_ID}\n\nLog: logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log\n\nGit commit: $(git log -1 --oneline)\n\nGit branch: $(git branch | grep "*")\n\nComment: ${comment}" > $output_dir/comment.txt
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $output_dir/log.log
    echo "$SLURM_JOB_ID" > $base_dir/latest.jobid
    ln -snf $output_dir $base_dir/latest.dir
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    echo "Node: $head_node"

    torchrun \
    --nnodes 1 \
    --nproc_per_node $num_gpus \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29522 \
        -m src.core.train \
            --do_train \
            --freeze_gate True \
            --eval_data_dir $eval_data_dir \
            --evaluation_strategy no \
            --dynamic_sampling_criterion mean \
            --run_name $task_name \
            --model_type $model_type \
            --model_name_or_path $model_name_or_path \
            --dataset_dir_or_path $dataset_dir_or_path \
            --output_dir $output_dir \
            --deepspeed conf/ds_bf16_zero1.json \
            --bf16 True \
            --tf32 True \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --max_steps 2000 \
            --save_strategy steps \
            --save_steps 9999999999999 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --report_to tensorboard

    note "Start MT-Bench inference for ${task_name}, wait for devices"
    python -m src.utils.gpu $CUDA_VISIBLE_DEVICES
    note "MT-Bench inference ${task_name} device OK - ${CUDA_VISIBLE_DEVICES}"
    python -m src.eval.gen_mt_ans --model-id $task_name --model-path $output_dir

    mbpp $task_name $output_dir
    multi_eval $task_name $output_dir

}

{
    # for task_num in [5, 10, 15, 20, 25, 30], run task job
    for task_num in 5 10 15 20 25 30
    do
        task_num_job $task_num
    done
}
