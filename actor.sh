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

python3 ../../actor.py robot_type@_global_=a2d task@_global_=${task_name} classifier_cfg.require_train=true use_human_intervention=true ego_mode=false policy_type=silri

# [debug]
# python3 ../../actor.py robot_type@_global_=a2d task@_global_=${task_name} classifier_cfg.require_train=true freeze_actor=true use_human_intervention=false ego_mode=false policy_type=silri