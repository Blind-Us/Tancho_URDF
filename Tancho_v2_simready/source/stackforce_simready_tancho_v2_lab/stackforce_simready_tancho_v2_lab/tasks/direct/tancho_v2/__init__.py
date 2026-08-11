import gymnasium as gym
from . import agents

gym.register(
    id="Template-TanchoV2-Direct-v0",
    # 1. 關鍵修改：Manager-Based 環境直接指向 Isaac Lab 原生執行器
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 2. 關鍵修改：對齊 tancho_v2_env_cfg.py 中的類別名稱 PTanchoV2EnvCfg
        "env_cfg_entry_point": f"{__name__}.tancho_v2_env_cfg:PTanchoV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TanchoV2PPORunnerCfg",
    },
)