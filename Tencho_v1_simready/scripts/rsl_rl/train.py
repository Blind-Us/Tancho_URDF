"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run distributed training.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import inspect
import logging
import os
import platform
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
try:
    from rsl_rl.algorithms import PPO
except ImportError:
    PPO = None
try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import stackforce_simready_tencho_v1_lab.tasks  # noqa: F401

logger = logging.getLogger(__name__)

import os
import torch


class _StackForceOnnxPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        actions = self.policy(obs)
        if isinstance(actions, (tuple, list)):
            return actions[0]
        if isinstance(actions, dict):
            if "actions" in actions:
                return actions["actions"]
            if "action" in actions:
                return actions["action"]
            return next(iter(actions.values()))
        return actions


def _stackforce_policy_obs_tensor(obs):
    if isinstance(obs, dict):
        obs = obs["policy"] if "policy" in obs else next(iter(obs.values()))
    elif not isinstance(obs, torch.Tensor) and hasattr(obs, "get"):
        try:
            candidate = obs.get("policy")
        except Exception:
            candidate = None
        if candidate is not None:
            obs = candidate
    if not isinstance(obs, torch.Tensor):
        raise TypeError(f"ONNX export requires a tensor policy observation, got {type(obs)!r}")
    if hasattr(obs, "detach"):
        obs = obs.detach()
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    elif obs.shape[0] > 1:
        obs = obs[:1]
    return obs.contiguous()


def stackforce_export_policy_as_onnx(policy, obs, output_dir, file_name="policy.onnx", opset=17):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    sample_obs = _stackforce_policy_obs_tensor(obs)
    module = _StackForceOnnxPolicy(policy).to(sample_obs.device).eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            sample_obs,
            output_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=opset,
        )
    print(f"Exported ONNX policy to: {output_path}")
    return output_path


try:
    from rsl_rl.runners import DistillationRunner
except ImportError:
    DistillationRunner = None

RSL_RL_VERSION = "3.0.1"
try:
    installed_version = metadata.version("rsl-rl-lib")
except metadata.PackageNotFoundError:
    installed_version = None

if installed_version is not None and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL. Existing version is: '{installed_version}' and required version is: '{RSL_RL_VERSION}'.\n"
        f"To install the correct version, run:\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _runner_uses_obs_groups():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return "resolve_obs_groups" in source or '"obs_groups"' in source or "'obs_groups'" in source


def _runner_uses_split_actor_critic():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return 'cfg["actor"]' in source or "cfg['actor']" in source or 'cfg["critic"]' in source or "cfg['critic']" in source


def _runner_expects_privileged_step():
    try:
        source = inspect.getsource(OnPolicyRunner.learn)
    except OSError:
        return True
    return "privileged_obs" in source or "critic_obs" in source


def _format_rsl_rl_obs(obs_dict, use_obs_groups):
    if not use_obs_groups:
        return obs_dict["policy"]
    if TensorDict is not None and not isinstance(obs_dict, TensorDict):
        first_obs = next(iter(obs_dict.values()))
        return TensorDict(dict(obs_dict), batch_size=[first_obs.shape[0]], device=first_obs.device)
    return obs_dict


class LegacyRslRlVecEnvWrapper(VecEnv):
    def __init__(self, env, clip_actions=None):
        self.env = env
        self.clip_actions = clip_actions
        self.use_obs_groups = _runner_uses_obs_groups()
        self.return_privileged_obs = _runner_expects_privileged_step()
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.cfg = env.unwrapped.cfg
        self.num_actions = gym.spaces.flatdim(env.unwrapped.single_action_space)
        obs_dict, extras = self.env.reset()
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.num_obs = obs_dict["policy"].shape[-1]
        self.num_privileged_obs = self.privileged_obs_buf.shape[-1] if self.privileged_obs_buf is not None else None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = env.unwrapped.episode_length_buf
        self.extras = extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self, env_ids=None):
        del env_ids
        obs_dict, extras = self.env.reset()
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.extras = extras
        if not self.return_privileged_obs:
            return self.obs_buf
        return self.obs_buf, self.privileged_obs_buf

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.env.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        episode_log = dict(extras.get("episode", {}))
        if torch.any(dones.bool()) and "log" in extras:
            episode_log.update(extras["log"])
        episode_log["Step_Reward/mean"] = torch.mean(rewards.detach())
        episode_log["Step_Reward/abs_mean"] = torch.mean(torch.abs(rewards.detach()))
        extras["episode"] = episode_log
        self.obs_buf = _format_rsl_rl_obs(obs_dict, self.use_obs_groups)
        self.privileged_obs_buf = obs_dict.get("critic")
        self.rew_buf = rewards
        self.reset_buf = dones
        self.extras = extras
        if not self.return_privileged_obs:
            return self.obs_buf, rewards, dones, extras
        return self.obs_buf, self.privileged_obs_buf, rewards, dones, extras

    def close(self):
        return self.env.close()


def _runner_expects_nested_runner():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return True
    return 'train_cfg["runner"]' in source or "train_cfg['runner']" in source


def _runner_uses_nested_class_name():
    try:
        source = inspect.getsource(OnPolicyRunner)
    except OSError:
        return False
    return (
        'algorithm"]["class_name' in source
        or "algorithm']['class_name" in source
        or 'policy_cfg.pop("class_name")' in source
        or "policy_cfg.pop('class_name')" in source
        or 'self.policy_cfg.pop("class_name")' in source
        or "self.policy_cfg.pop('class_name')" in source
        or "resolve_callable" in source
    )


def _runner_uses_split_actor_critic():
    try:
        source = inspect.getsource(OnPolicyRunner)
        if PPO is not None and hasattr(PPO, "construct_algorithm"):
            source += "\n" + inspect.getsource(PPO.construct_algorithm)
    except OSError:
        return False
    return 'cfg["actor"]' in source or "cfg['actor']" in source or 'cfg["critic"]' in source or "cfg['critic']" in source


def to_compatible_rsl_rl_cfg(agent_cfg):
    data = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    allowed_policy_keys = {
        "actor_hidden_dims",
        "critic_hidden_dims",
        "activation",
        "init_noise_std",
        "clip_actions",
        "actor_obs_normalization",
        "critic_obs_normalization",
    }
    allowed_algorithm_keys = {
        "num_learning_epochs",
        "num_mini_batches",
        "clip_param",
        "gamma",
        "lam",
        "value_loss_coef",
        "entropy_coef",
        "learning_rate",
        "max_grad_norm",
        "use_clipped_value_loss",
        "schedule",
        "desired_kl",
        "use_spo",
    }
    if "runner" in data and "policy" in data and "algorithm" in data:
        runner_cfg = dict(data["runner"])
        policy_cfg = {key: value for key, value in dict(data["policy"]).items() if key in allowed_policy_keys or key == "class_name"}
        algorithm_cfg = {key: value for key, value in dict(data["algorithm"]).items() if key in allowed_algorithm_keys or key == "class_name"}
    else:
        policy_cfg = {key: value for key, value in dict(data["policy"]).items() if key in allowed_policy_keys}
        algorithm_cfg = {key: value for key, value in dict(data["algorithm"]).items() if key in allowed_algorithm_keys}
        runner_cfg = {key: value for key, value in data.items() if key not in {"policy", "algorithm", "class_name"}}

    runner_cfg.setdefault("num_steps_per_env", getattr(agent_cfg, "num_steps_per_env", 24))
    runner_cfg.setdefault("max_iterations", getattr(agent_cfg, "max_iterations", 1500))
    runner_cfg.setdefault("save_interval", getattr(agent_cfg, "save_interval", 50))
    runner_cfg.setdefault("obs_groups", {"policy": ["policy"], "critic": ["policy"]})
    runner_cfg.setdefault("experiment_name", getattr(agent_cfg, "experiment_name", "stackforce"))
    runner_cfg.setdefault("run_name", getattr(agent_cfg, "run_name", ""))
    runner_cfg.setdefault("resume", getattr(agent_cfg, "resume", False))
    runner_cfg.setdefault("load_run", getattr(agent_cfg, "load_run", ".*"))
    runner_cfg.setdefault("checkpoint", getattr(agent_cfg, "load_checkpoint", "model_.*.pt"))

    if _runner_uses_nested_class_name():
        policy_cfg.setdefault("class_name", "ActorCritic")
        algorithm_cfg.setdefault("class_name", "PPO")
    else:
        runner_cfg.setdefault("policy_class_name", "ActorCritic")
        runner_cfg.setdefault("algorithm_class_name", "PPO")

    if _runner_expects_nested_runner():
        return {"runner": runner_cfg, "policy": policy_cfg, "algorithm": algorithm_cfg}
    if _runner_uses_split_actor_critic():
        algorithm_cfg.setdefault("class_name", "PPO")
        algorithm_cfg.pop("use_spo", None)
        actor_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [256, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy_cfg.get("init_noise_std", 1.0),
            },
        }
        critic_cfg = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [256, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }
        runner_cfg.pop("policy_class_name", None)
        runner_cfg.pop("algorithm_class_name", None)
        runner_cfg["obs_groups"] = {"actor": ["policy"], "critic": ["policy"], "policy": ["policy"]}
        runner_cfg.setdefault("multi_gpu", None)
        return {**runner_cfg, "actor": actor_cfg, "critic": critic_cfg, "algorithm": algorithm_cfg}
    return {**runner_cfg, "policy": policy_cfg, "algorithm": algorithm_cfg}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported on CPU.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments.")

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = LegacyRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    legacy_agent_cfg = to_compatible_rsl_rl_cfg(agent_cfg)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, legacy_agent_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner" and DistillationRunner is not None:
        runner = DistillationRunner(env, legacy_agent_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    if hasattr(runner, "add_git_repo_to_log"):
        runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"Saved final checkpoint to: {final_path}")
    try:
        stackforce_export_policy_as_onnx(
            runner.get_inference_policy(device=env.device),
            env.get_observations(),
            os.path.join(log_dir, "exported", "policies"),
        )
    except Exception as exc:
        print(f"ONNX export skipped: {exc}")
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
