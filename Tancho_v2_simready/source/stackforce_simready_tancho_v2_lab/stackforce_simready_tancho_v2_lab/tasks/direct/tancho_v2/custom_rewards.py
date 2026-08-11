from __future__ import annotations

import torch


def compute_custom_reward_terms(env) -> dict[str, torch.Tensor]:
    """Return additional unscaled reward terms.

    Edit this file after export to add task-specific reward terms.
    Each tensor must have shape [num_envs].

    To activate a term, also set a non-zero scale for the same key
    inside reward_scales in the generated *_env_cfg.py file.
    """

    # Example:
    # root_height = env._robot.data.root_pos_w[:, 2] - env._terrain.env_origins[:, 2]
    # return {"custom_reward": torch.clamp(root_height - 0.25, min=0.0)}

    return {"custom_reward": torch.zeros(env.num_envs, dtype=torch.float, device=env.device)}
