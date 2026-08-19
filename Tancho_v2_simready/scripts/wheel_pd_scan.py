"""Scan simple pitch-feedback wheel controllers to verify actuator controllability."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Template-TanchoV2-Direct-v0")
parser.add_argument("--max_steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import stackforce_simready_tancho_v2_lab.tasks  # noqa: F401


def main():
    # (wheel pattern, proportional sign/gain, derivative gain)
    controllers = [
        (1.0, 1.0, 0.2), (1.0, -1.0, -0.2),
        (1.0, 3.0, 0.5), (1.0, -3.0, -0.5),
        (-1.0, 1.0, 0.2), (-1.0, -1.0, -0.2),
        (-1.0, 3.0, 0.5), (-1.0, -3.0, -0.5),
    ]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(controllers))
    env_cfg.seed = 42
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    episode_steps = torch.zeros(len(controllers), dtype=torch.long, device=unwrapped.device)
    completed = torch.zeros_like(episode_steps)
    step_sum = torch.zeros_like(episode_steps)
    max_survival = torch.zeros_like(episode_steps)

    pattern = torch.tensor([c[0] for c in controllers], device=unwrapped.device)
    kp = torch.tensor([c[1] for c in controllers], device=unwrapped.device)
    kd = torch.tensor([c[2] for c in controllers], device=unwrapped.device)
    for _ in range(args_cli.max_steps):
        gravity_x = robot.data.projected_gravity_b[:, 0]
        pitch_rate = robot.data.root_ang_vel_b[:, 1]
        command = torch.clamp(kp * gravity_x + kd * pitch_rate, -1.0, 1.0)
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        actions[:, 4] = command
        actions[:, 5] = pattern * command
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        episode_steps += 1
        done = terminated | truncated
        step_sum[done] += episode_steps[done]
        max_survival[done] = torch.maximum(max_survival[done], episode_steps[done])
        completed[done] += 1
        episode_steps[done] = 0

    max_survival = torch.maximum(max_survival, episode_steps)
    for i, (wheel_pattern, p_gain, d_gain) in enumerate(controllers):
        mean = float(step_sum[i]) / int(completed[i]) if completed[i] else float(episode_steps[i])
        print(
            f"WHEEL_PD_SCAN pattern={wheel_pattern:+.0f} kp={p_gain:+.1f} kd={d_gain:+.1f} "
            f"completed={int(completed[i])} mean_steps={mean:.2f} max_steps={int(max_survival[i])} "
            f"seconds={mean * float(unwrapped.step_dt):.3f}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
