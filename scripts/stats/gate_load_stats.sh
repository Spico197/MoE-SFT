#!/usr/bin/bash

#SBATCH --job-name=llama_moe
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --nodes=1
##SBATCH --gres=gpu:0
#SBATCH --quotatype=auto

export CUDA_VISIBLE_DEVICES=7
export WANDB_PROJECT="adaptive-moe-sft-stats"
num_gpus=1

{
    task_name="llama_moe_uniform"
    model_type="auto"
    model_name_or_path="./data/llama-moe-models/LLaMA-MoE-v1-3_5B-2_8-new"
    dataset_dir_or_path="data/merged_splits/split_7/fschat_0.jsonl"
    eval_data_dir="data/merged_splits_gate_load_results"

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

    # srun torchrun \
    # --nnodes 1 \
    # --nproc_per_node $num_gpus \
    # --node_rank $SLURM_NODEID \
    # --rdzv_id $RANDOM \
    # --rdzv_backend c10d \
    # --rdzv_endpoint $head_node:29522 \
    python \
        -m src.core.gate_load_stats \
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
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 1 \
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
            --report_to wandb

}
