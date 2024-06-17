set -x

multi_eval() {
    task_name=$1
    model_path=$2
    nohup srun -p MoE --gres gpu:1 bash scripts/eval/eval.sh qa $model_path True results/$task_name 1>logs/eval-$task_name-qa.log 2>&1 &
    sleep 1
    nohup srun -p MoE --gres gpu:1 bash scripts/eval/eval.sh bbh $model_path True results/$task_name 1>logs/eval-$task_name-bbh.log 2>&1 &
    sleep 1
    nohup srun -p MoE --gres gpu:1 bash scripts/eval/eval.sh reasoning $model_path True results/$task_name 1>logs/eval-$task_name-reasoning.log 2>&1 &
    sleep 1
    nohup srun -p MoE --gres gpu:1 bash scripts/eval/eval.sh mmlu $model_path True results/$task_name 1>logs/eval-$task_name-mmlu.log 2>&1 &
    sleep 1
    sbatch scripts/eval/code.sh mbpp $task_name $model_path
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
single_eval reasoning moduleformer_random outputs/moduleformer_random/2533914/
multi_eval moduleformer_random outputs/moduleformer_random/2533914/

}
