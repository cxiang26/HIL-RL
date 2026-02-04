#!/usr/bin/env python
"""
真机推理测试脚本：加载指定 policy 与（可选）classifier 权重，在真实机器人上跑策略。
不连接 Learner，不收发任何训练数据。

Usage:
    # 仅指定 policy（无分类器奖励）
    python run_inference.py --policy_path experiments/a2d_griper/exp_local/.../checkpoints/005900/pretrained_model

    # 同时指定 policy 与 classifier
    python run_inference.py \
        --policy_path experiments/a2d_griper/exp_local/.../checkpoints/005900/pretrained_model \
        --classifier_path experiments/a2d_griper/classifier/checkpoints/000500

    # 指定任务与 episode 数
    python run_inference.py --policy_path /path/to/policy --task a2d_left_right --num_episodes 5
"""

import argparse
import os
import sys

import numpy as np
import torch

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lerobot", "src"))

from hydra import compose, initialize
from omegaconf import OmegaConf
import draccus
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
# 以下导入用于向 draccus 注册 choice 类型，解析 train_config_silri_a2d.json 时必需（与 actor.py/learner.py 保持一致）
from lerobot import envs  # noqa: F401  env.type=gym_manipulator
from lerobot.cameras import opencv  # noqa: F401  env.robot.cameras.*.type=opencv
from lerobot.robots import so100_follower  # noqa: F401  env.robot.type=so100_follower_end_effector
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401  env.teleop.type=gamepad
from lerobot.policies.silri.configuration_silri import SiLRIConfig  # noqa: F401  policy.type=silri
from lerobot.utils.utils import get_safe_torch_device

from make_env import make_env


def make_policy_obs(obs: dict, device: torch.device, robot_type: str) -> dict:
    """将 env 的 obs 转为 policy 输入的 batch 格式。"""
    policy_obs = {}
    for key in obs.keys():
        if "state" not in key:
            img = torch.from_numpy(obs[key]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            new_key = "observation.images." + key
            policy_obs[new_key] = img
        else:
            state = torch.from_numpy(obs[key]).float().unsqueeze(0).to(device)
            policy_obs["observation.state"] = state
    return policy_obs


def main():
    parser = argparse.ArgumentParser(description="真机推理：指定 policy 与（可选）classifier 权重运行")
    parser.add_argument("--policy_path", type=str, required=True, help="策略权重目录（含 config.json 与 model.safetensors）")
    parser.add_argument("--classifier_path", type=str, default=None, help="分类器 checkpoint 目录（其下应有 pretrained_model），不指定则不使用分类器奖励")
    parser.add_argument("--task", type=str, default="a2d_griper", help="任务名，对应 cfg/task/<task>.yaml")
    parser.add_argument("--robot_type", type=str, default="a2d", help="机器人类型")
    parser.add_argument("--num_episodes", type=int, default=10, help="运行 episode 数")
    parser.add_argument("--max_steps", type=int, default=0, help="每 episode 最大步数，0 表示使用任务配置的 max_episode_length")
    parser.add_argument("--device", type=str, default="cuda", help="policy 运行设备")
    args = parser.parse_args()

    if not os.path.isdir(args.policy_path):
        print(f"Error: policy_path 不是目录或不存在: {args.policy_path}")
        sys.exit(1)
    if args.classifier_path is not None and not os.path.isdir(args.classifier_path):
        print(f"Error: classifier_path 不是目录或不存在: {args.classifier_path}")
        sys.exit(1)

    # 1. Hydra 加载环境配置
    with initialize(config_path="cfg", version_base=None):
        env_cfg = compose(
            config_name="config",
            overrides=[
                f"robot_type@_global_={args.robot_type}",
                f"task@_global_={args.task}",
                "fake_env=false",
            ],
        )

    # 若指定了 classifier，覆盖 classifier_cfg 并确保 load_classifier=True
    if args.classifier_path:
        if not hasattr(env_cfg.robot_config, "classifier_cfg"):
            env_cfg.robot_config["classifier_cfg"] = OmegaConf.create({})
        env_cfg.robot_config.classifier_cfg.checkpoint_path = args.classifier_path
        env_cfg.robot_config.classifier_cfg.load_classifier = True
        env_cfg.robot_config.classifier_cfg.require_train = False

    # 2. 加载 lerobot 训练配置并设置 policy 路径与 state 维度
    lerobot_config_path = os.path.join(PROJECT_ROOT, "train_config_silri_a2d.json")
    if not os.path.isfile(lerobot_config_path):
        print(f"Error: 未找到 {lerobot_config_path}")
        sys.exit(1)

    with draccus.config_type("json"):
        cfg = draccus.parse(TrainRLServerPipelineConfig, lerobot_config_path, args=["--policy.type=silri", "--policy.num_discrete_actions=2"])

    cfg.policy.pretrained_path = args.policy_path
    cfg.dataset = None
    if env_cfg.robot_config.robot_type == "a2d":
        cfg.env.features["observation.state"].shape = [8]
        cfg.policy.input_features["observation.state"].shape = [8]
    else:
        cfg.env.features["observation.state"].shape = [14] if getattr(env_cfg, "use_force", False) else [8]
        cfg.policy.input_features["observation.state"].shape = [14] if getattr(env_cfg, "use_force", False) else [8]

    # 3. 创建环境（可选挂载 classifier）
    env = make_env(
        config=env_cfg,
        fake_env=False,
        use_human_intervention=False,
        classifier=args.classifier_path is not None,
        use_gripper_penalty=getattr(cfg.policy, "use_gripper_penalty", False),
        cfg=cfg if args.classifier_path else None,
    )

    max_steps = args.max_steps if args.max_steps > 0 else env_cfg.robot_config.max_episode_length

    # 4. 创建 policy
    device = get_safe_torch_device(args.device, log=True)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.to(device)
    policy.eval()

    print("=" * 60)
    print("真机推理测试")
    print("=" * 60)
    print(f"  policy_path:      {args.policy_path}")
    print(f"  classifier_path: {args.classifier_path or '(未使用)'}")
    print(f"  task:            {args.task}")
    print(f"  robot_type:      {args.robot_type}")
    print(f"  num_episodes:    {args.num_episodes}")
    print(f"  max_steps/ep:    {max_steps}")
    print("=" * 60)

    episode_rewards = []
    for ep in range(args.num_episodes): 
        obs, info = env.reset()
        episode_reward = 0.0
        for step in range(max_steps):
            with torch.no_grad():
                policy_obs = make_policy_obs(obs, device, env_cfg.robot_config.robot_type)
                # 推理默认使用策略均值（deterministic=True），非重采样
                policy_action, _ = policy.select_action(batch=policy_obs, deterministic=True)
            action = policy_action.squeeze(0).cpu().numpy()
            if env_cfg.robot_config.robot_type != "sim":
                action_full = np.zeros(7)
                action_full[: len(action)] = action
                action = action_full

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated
            if step % 10 == 0 or done:
                print(f"  Episode {ep+1} step {step}  reward={reward:.3f}  done={done}")
            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1} 总奖励: {episode_reward:.3f}")

    env.close()
    print("=" * 60)
    print(f"完成 {len(episode_rewards)} 个 episode，平均奖励: {np.mean(episode_rewards):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
