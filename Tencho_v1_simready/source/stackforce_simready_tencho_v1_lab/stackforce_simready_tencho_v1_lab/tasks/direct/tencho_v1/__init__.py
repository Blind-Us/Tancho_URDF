import gymnasium as gym

from . import agents


gym.register(
    id="Template-TenchoV1-Direct-v0",
    entry_point=f"{__name__}.tencho_v1_env:TenchoV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tencho_v1_env_cfg:TenchoV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TenchoV1PPORunnerCfg",
    },
)
