#!/usr/bin/bash

#SBATCH --job-name=moduleformer_dynamic_ref_loss
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=MoE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --quotatype=auto

export WANDB_PROJECT="adaptive-moe-sft"
num_gpus=4

{
    task_name="moduleformer_dynamic_ref_loss"
    model_type="auto"
    model_name_or_path="/mnt/petrelfs/zhutong/llama-moe-models/MoLM-700M-4B"
    dataset_dir_or_path="data/four_types_mix/train"
    eval_data_dir="data/four_types_mix/dev"

    comment="MoLM-700M-4B, four type mix, dynamic baseline, 4 gpus, eval_steps 100, max_eval_steps 5"
    base_dir="outputs"
    output_dir="${base_dir}/${task_name}/$SLURM_JOB_ID"
    mkdir -p $output_dir
    scontrol write batch_script $SLURM_JOBID $output_dir/sbatch.sh
    git diff > $output_dir/diff.patch
    env > $output_dir/env
    echo -e "Job ID: ${SLURM_JOB_ID}\n\nLog: logs/moduleformer_dynamic_ref_loss-$SLURM_JOB_ID.log\n\nGit commit: $(git log -1 --oneline)\n\nGit branch: $(git branch | grep "*")\n\nComment: ${comment}" > $output_dir/comment.txt
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $output_dir/log.log
    echo "$SLURM_JOB_ID" > $base_dir/latest.jobid
    ln -snf $output_dir $base_dir/latest.dir
    ln -snf $(scontrol show job $SLURM_JOB_ID | grep "StdOut=" | cut -d '=' -f 2) $base_dir/latest.log

    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIS))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    echo "Node: $head_node"

    srun torchrun \
    --nnodes 1 \
    --nproc_per_node $num_gpus \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29522 \
        -m src.core.train \
            --do_train \
            --do_eval \
            --freeze_gate True \
            --dynamic_sampling_type loss \
            --dynamic_sampling_name2ref_loss conf/ref_loss/moduleformer_uniform.json \
            --dynamic_c 0.8 \
            --attn_impl eager \
            --eval_data_dir $eval_data_dir \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --max_eval_steps 5 \
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
            --report_to wandb

    python -m src.eval.gen_mt_ans \
        --model-path $output_dir \
        --model-id $task_name

    # python -m src.eval.gen_alpaca_eval_ans \
    #     --model-path $output_dir \
    #     --model-id $task_name
}
