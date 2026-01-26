
export PYTHONPATH=$PYTHONPATH:./lerobot/src/
export PYTHONPATH=$PYTHONPATH:./rl_envs/
export PYTHONPATH=$PYTHONPATH:../HIL-RL
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
# export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889


# 接收 task_name 参数，默认为 test_a2d_task
task_name=${1:-test_a2d_task}

echo "Task name: ${task_name}"
mkdir -p experiments/${task_name}
cd experiments/${task_name}

python3 ../../learner.py robot_type@_global_=a2d task@_global_=${task_name} policy_type=silri
