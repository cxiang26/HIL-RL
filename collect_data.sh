export PYTHONPATH=$PYTHONPATH:./lerobot/src/
export PYTHONPATH=$PYTHONPATH:./rl_envs/
export PYTHONPATH=$PYTHONPATH:../HIL-RL
# export PYTHONPATH=$PYTHONPATH:/home/eai/Dev/sysEAI/xRocs/xRocs
# export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889



# 接收 task_name 参数，默认为 a2d_griper
task_name=${1:-test_a2d_task}

echo "Task name: ${task_name}"
mkdir -p experiments/${task_name}
cd experiments/${task_name}


python3 ../../collect_data.py robot_type@_global_=a2d task@_global_=${task_name} use_human_intervention=true ego_mode=true load_classifier=false
