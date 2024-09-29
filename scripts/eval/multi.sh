set -x

multi_eval() {
    task_name=$1
    model_path=$2
    CUDA_VISIBLE_DEVICES=0 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh qa $model_path True results/$task_name 1>logs/eval-$task_name-qa.log 2>&1 &
    sleep 1
    CUDA_VISIBLE_DEVICES=1 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh bbh $model_path True results/$task_name 1>logs/eval-$task_name-bbh.log 2>&1 &
    sleep 1
    CUDA_VISIBLE_DEVICES=2 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh reasoning $model_path True results/$task_name 1>logs/eval-$task_name-reasoning.log 2>&1 &
    sleep 1
    CUDA_VISIBLE_DEVICES=3 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh mmlu $model_path True results/$task_name 1>logs/eval-$task_name-mmlu.log 2>&1 &

    # CUDA_VISIBLE_DEVICES=4 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh qa $model_path True results/$task_name 1>logs/eval-$task_name-qa.log 2>&1 &
    # sleep 1
    # CUDA_VISIBLE_DEVICES=5 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh bbh $model_path True results/$task_name 1>logs/eval-$task_name-bbh.log 2>&1 &
    # sleep 1
    # CUDA_VISIBLE_DEVICES=6 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh reasoning $model_path True results/$task_name 1>logs/eval-$task_name-reasoning.log 2>&1 &
    # sleep 1
    # CUDA_VISIBLE_DEVICES=7 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh mmlu $model_path True results/$task_name 1>logs/eval-$task_name-mmlu.log 2>&1 &

    # sleep 1
    # sbatch scripts/eval/code.sh mbpp $task_name $model_path
}


multi_eval2() {
    task_name=$1
    model_path=$2
    CUDA_VISIBLE_DEVICES=4 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh qa $model_path True results/$task_name 1>logs/eval-$task_name-qa.log 2>&1 &
    sleep 1
    CUDA_VISIBLE_DEVICES=5 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh bbh $model_path True results/$task_name 1>logs/eval-$task_name-bbh.log 2>&1 &
    sleep 1
    CUDA_VISIBLE_DEVICES=6 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh reasoning $model_path True results/$task_name 1>logs/eval-$task_name-reasoning.log 2>&1 &
    sleep 1
    CUDA_VISIBLE_DEVICES=7 nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh mmlu $model_path True results/$task_name 1>logs/eval-$task_name-mmlu.log 2>&1 &

    # sleep 1
    # sbatch scripts/eval/code.sh mbpp $task_name $model_path
}

single_eval() {
    task=$1
    run_name=$2
    model_path=$3

    if [ $task = "mbpp" ] || [ $task = "humaneval" ]; then
        sbatch scripts/eval/code.sh $task $run_name $model_path
    else
        nohup srun -p MoE --gres gpu:1 bash scripts/eval/eval.sh $task $model_path True results/$run_name 1>logs/eval-$run_name-$task.log 2>&1 &
    fi
}

seval() {
    device=$1
    task=$2
    run_name=$3
    model_path=$4

    if [ $task = "mbpp" ] || [ $task = "humaneval" ]; then
        sbatch scripts/eval/code.sh $task $run_name $model_path
    else
        CUDA_VISIBLE_DEVICES=$device nohup srun -p MoE -w SH-IDCA1404-10-140-54-12 bash scripts/eval/eval.sh $task $model_path True results/$run_name 1>logs/eval-$run_name-$task.log 2>&1 &
    fi
    sleep 1
}

listen_eval() {
    task_name=$1
    model_path=$2
    shift 2
    nohup python -m src.eval.listen $task_name $model_path $* 1>logs/listen_eval-$task_name.log 2>&1 &
}

gen_one() {
    task=$1
    run_name=$2
    model_path=$3
    shift 3
    sbatch scripts/eval/gen.sh $task $run_name $model_path $*
}

{

# e.g.
# single_eval reasoning moduleformer_random outputs/moduleformer_random/2533914/
# multi_eval moduleformer_random outputs/moduleformer_random/2533914/

# multi_eval balancing_0_0.0125 outputs/balancing_0_0.0125/3617244
# multi_eval2 balancing_0.0125_0.025 outputs/balancing_0.0125_0.025/3620620
# multi_eval balancing_0.025_0.0375 outputs/balancing_0.025_0.0375/3620621
# multi_eval balancing_0.0375_0.05 outputs/balancing_0.0375_0.05/3622879
# multi_eval balancing_0.05_0.0625 outputs/balancing_0.05_0.0625/3622880
# multi_eval balancing_0.0625_0.075 outputs/balancing_0.0625_0.075/3624909
# multi_eval balancing_0.075_0.0875 outputs/balancing_0.075_0.0875/3624911
# multi_eval balancing_0.0875_0.1 outputs/balancing_0.0875_0.1/3617243

# seval 0 qa balancing_0_0.0125 outputs/balancing_0_0.0125/3617244
# seval 1 qa balancing_0.0125_0.025 outputs/balancing_0.0125_0.025/3620620
# seval 2 qa balancing_0.025_0.0375 outputs/balancing_0.025_0.0375/3620621
# seval 3 qa balancing_0.0375_0.05 outputs/balancing_0.0375_0.05/3622879
# seval 4 qa balancing_0.05_0.0625 outputs/balancing_0.05_0.0625/3622880
# seval 5 qa balancing_0.0625_0.075 outputs/balancing_0.0625_0.075/3624909
# seval 6 qa balancing_0.075_0.0875 outputs/balancing_0.075_0.0875/3624911
# seval 7 qa balancing_0.0875_0.1 outputs/balancing_0.0875_0.1/3617243

# seval 0 mmlu balancing_0_0.0125 outputs/balancing_0_0.0125/3617244
# seval 1 mmlu balancing_0.0125_0.025 outputs/balancing_0.0125_0.025/3620620
# seval 2 mmlu balancing_0.025_0.0375 outputs/balancing_0.025_0.0375/3620621
# seval 3 mmlu balancing_0.0375_0.05 outputs/balancing_0.0375_0.05/3622879
# seval 4 mmlu balancing_0.05_0.0625 outputs/balancing_0.05_0.0625/3622880
# seval 5 mmlu balancing_0.0625_0.075 outputs/balancing_0.0625_0.075/3624909
# seval 6 mmlu balancing_0.075_0.0875 outputs/balancing_0.075_0.0875/3624911
# seval 7 mmlu balancing_0.0875_0.1 outputs/balancing_0.0875_0.1/3617243

# seval 0 mmlu balancing_0_0.0125 outputs/balancing_0_0.0125/3617244
# seval 1 qa balancing_0.0375_0.05 outputs/balancing_0.0375_0.05/3622879
# seval 2 mmlu balancing_0.05_0.0625 outputs/balancing_0.05_0.0625/3622880

# seval 0 reasoning balancing_0_0.0125 outputs/balancing_0_0.0125/3617244
# seval 1 reasoning balancing_0.0125_0.025 outputs/balancing_0.0125_0.025/3620620
# seval 2 reasoning balancing_0.025_0.0375 outputs/balancing_0.025_0.0375/3620621
# seval 3 reasoning balancing_0.0375_0.05 outputs/balancing_0.0375_0.05/3622879
# seval 4 reasoning balancing_0.05_0.0625 outputs/balancing_0.05_0.0625/3622880
# seval 5 reasoning balancing_0.0625_0.075 outputs/balancing_0.0625_0.075/3624909
# seval 6 reasoning balancing_0.075_0.0875 outputs/balancing_0.075_0.0875/3624911
# seval 7 reasoning balancing_0.0875_0.1 outputs/balancing_0.0875_0.1/3617243

# seval 0 bbh balancing_0_0.0125 outputs/balancing_0_0.0125/3617244
# seval 1 bbh balancing_0.0125_0.025 outputs/balancing_0.0125_0.025/3620620
# seval 2 bbh balancing_0.025_0.0375 outputs/balancing_0.025_0.0375/3620621
# seval 3 bbh balancing_0.0375_0.05 outputs/balancing_0.0375_0.05/3622879
# seval 4 bbh balancing_0.05_0.0625 outputs/balancing_0.05_0.0625/3622880
# seval 5 bbh balancing_0.0625_0.075 outputs/balancing_0.0625_0.075/3624909
# seval 6 bbh balancing_0.075_0.0875 outputs/balancing_0.075_0.0875/3624911
# seval 7 bbh balancing_0.0875_0.1 outputs/balancing_0.0875_0.1/3617243


export HF_DATASETS_OFFLINE=1

# seval 0 qa olmoe_wo_sft data/llama-moe-models/OLMoE-1B-7B-0924
# seval 1 qa olmoe_datasize outputs/olmoe_datasize/olmoe_datasize-3656417
# seval 2 qa olmoe_uniform outputs/olmoe_uniform/3647027
# seval 3 qa olmoe_random outputs/olmoe_random/3668835
# seval 4 qa olmoe_sequential outputs/olmoe_sequential/3665915
# seval 5 qa olmoe_refloss outputs/olmoe_refloss/3670750
# seval 6 qa olmoe_dynamic outputs/olmoe_dynamic/3668915

# seval 7 mmlu olmoe_wo_sft data/llama-moe-models/OLMoE-1B-7B-0924
# seval 1 mmlu olmoe_datasize outputs/olmoe_datasize/olmoe_datasize-3656417
# seval 2 mmlu olmoe_uniform outputs/olmoe_uniform/3647027
# seval 3 mmlu olmoe_random outputs/olmoe_random/3668835
# seval 4 mmlu olmoe_sequential outputs/olmoe_sequential/3665915
# seval 5 mmlu olmoe_refloss outputs/olmoe_refloss/3670750
# seval 6 mmlu olmoe_dynamic outputs/olmoe_dynamic/3668915

# seval 0 bbh olmoe_wo_sft data/llama-moe-models/OLMoE-1B-7B-0924
# seval 7 bbh olmoe_datasize outputs/olmoe_datasize/olmoe_datasize-3656417
# seval 0 bbh olmoe_uniform outputs/olmoe_uniform/3647027
# seval 1 bbh olmoe_random outputs/olmoe_random/3668835
# seval 2 bbh olmoe_sequential outputs/olmoe_sequential/3665915
# seval 3 bbh olmoe_refloss outputs/olmoe_refloss/3670750
# seval 4 bbh olmoe_dynamic outputs/olmoe_dynamic/3668915

# seval 5 reasoning olmoe_wo_sft data/llama-moe-models/OLMoE-1B-7B-0924
# seval 6 reasoning olmoe_datasize outputs/olmoe_datasize/olmoe_datasize-3656417
# seval 7 reasoning olmoe_uniform outputs/olmoe_uniform/3647027
# seval 0 reasoning olmoe_random outputs/olmoe_random/3668835
# seval 1 reasoning olmoe_sequential outputs/olmoe_sequential/3665915
# seval 2 reasoning olmoe_refloss outputs/olmoe_refloss/3670750
# seval 3 reasoning olmoe_dynamic outputs/olmoe_dynamic/3668915

# seval 4,5,6,7 mbpp olmoe_wo_sft data/llama-moe-models/OLMoE-1B-7B-0924
# seval 0,1,2,3 mbpp olmoe_datasize outputs/olmoe_datasize/olmoe_datasize-3656417
# seval 4,5,6,7 mbpp olmoe_uniform outputs/olmoe_uniform/3647027
# seval 0,1,2,3 mbpp olmoe_random outputs/olmoe_random/3668835
# seval 4,5,6,7 mbpp olmoe_sequential outputs/olmoe_sequential/3665915
# seval 0,1,2,3 mbpp olmoe_refloss outputs/olmoe_refloss/3670750
# seval 4,5,6,7 mbpp olmoe_dynamic outputs/olmoe_dynamic/3668915


# seval 0 qa olmoe_dynamic_m200 outputs/olmoe_dynamic_m200/3673738
# seval 1 mmlu olmoe_dynamic_m200 outputs/olmoe_dynamic_m200/3673738

# seval 0 qa olmoe_dynamic_m200_c0.2 outputs/olmoe_dynamic_m200_c0.2/3675353
# seval 1 mmlu olmoe_dynamic_m200_c0.2 outputs/olmoe_dynamic_m200_c0.2/3675353

# seval 2 qa olmoe_dynamic_m200_c0.1 outputs/olmoe_dynamic_m200_c0.1/3675354
# seval 3 mmlu olmoe_dynamic_m200_c0.1 outputs/olmoe_dynamic_m200_c0.1/3675354

seval 0 qa olmoe_dynamic_m200_c0.4 outputs/olmoe_dynamic_m200_c0.4/3676913
seval 1 mmlu olmoe_dynamic_m200_c0.4 outputs/olmoe_dynamic_m200_c0.4/3676913


}
