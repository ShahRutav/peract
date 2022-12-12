EXP_DIR=$1
if [ -z "${EXP_DIR}" ]; then
    echo "Please pass the framework.logdir from the specified in train config file"
else
    echo $EXP_DIR
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        rlbench.tasks=[open_drawer] \
        rlbench.task_name='single_1k' \
        rlbench.demo_path=/home/rutavms/data/peract/val \
        framework.logdir=$EXP_DIR \
        framework.csv_logging=True \
        framework.tensorboard_logging=True \
        framework.eval_envs=1 \
        framework.start_seed=0 \
        framework.eval_from_eps_number=0 \
        framework.eval_episodes=25 \
        framework.eval_type='missing' \
        method.name=VIT_BC_LANG \
        rlbench.headless=True
fi
