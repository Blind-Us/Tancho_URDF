"""Measure passive survival for several thigh targets in parallel."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Flat-v0")
parser.add_argument("--max_steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import tancho_v3_lab.tasks  # noqa: F401


def main():
    thigh_targets = torch.tensor(
        [-0.70, -0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35], device=args_cli.device
        #[-0.50, -0.44, -0.38, -0.32, -0.26, -0.20, -0.14, -0.08], device=args_cli.device
    )
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(thigh_targets))
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    episode_steps = torch.zeros(len(thigh_targets), dtype=torch.long, device=unwrapped.device)
    completed = torch.zeros_like(episode_steps)
    step_sum = torch.zeros_like(episode_steps)
    max_survival = torch.zeros_like(episode_steps)

    joint_pos_term = unwrapped.action_manager.get_term("joint_pos")
    print(
        f"POSE_SCAN_ACTION_MAP robot_joint_names={unwrapped.scene['robot'].joint_names} "
        f"term_joint_ids={joint_pos_term._joint_ids}",
        flush=True,
    )
    # JointPositionActionCfg resolves the requested names to articulation order:
    # L thigh, R thigh, L calf, R calf.
    thigh_action = (thigh_targets - float(env_cfg.scene.robot.init_state.joint_pos["joint_thigh_L"])) / float(
        env_cfg.actions.joint_pos.scale
    )
    for _ in range(args_cli.max_steps):
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        actions[:, 0] = thigh_action
        actions[:, 1] = thigh_action
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        episode_steps += 1
        done = terminated | truncated
        step_sum[done] += episode_steps[done]
        max_survival[done] = torch.maximum(max_survival[done], episode_steps[done])
        completed[done] += 1
        episode_steps[done] = 0

    max_survival = torch.maximum(max_survival, episode_steps)
    for i, target in enumerate(thigh_targets.tolist()):
        mean = float(step_sum[i]) / int(completed[i]) if completed[i] else float(episode_steps[i])
        print(
            f"POSE_SCAN thigh={target:.3f} completed={int(completed[i])} "
            f"mean_steps={mean:.2f} max_steps={int(max_survival[i])} seconds={mean * float(unwrapped.step_dt):.3f}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
