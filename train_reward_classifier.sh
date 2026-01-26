export PYTHONPATH=$PYTHONPATH:./lerobot/src/
export PYTHONPATH=$PYTHONPATH:./rl_envs/
export PYTHONPATH=$PYTHONPATH:../HIL-RL
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
# 如果不需要代理，可以注释掉下面这行
# export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889
# 如果代理无法连接，可以禁用代理或设置 HuggingFace 不使用代理
# export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
# unset http_proxy https_proxy



# 接收 task_name 参数，默认为 test_a2d_task
task_name=${1:-test_a2d_task}

echo "Task name: ${task_name}"
mkdir -p experiments/${task_name}
cd experiments/${task_name}


python3 ../../train_reward_classifier.py \
    --config_path ../../train_config_reward_classifier.json \
    --dataset.root="../../experiments/${task_name}/offline_dataset" \
    "${@:2}"  