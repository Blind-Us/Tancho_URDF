# Tancho V3 Isaac Lab Training Project

This project uses the standalone `tancho_v3_lab` package and is ready to train in Isaac Lab / Isaac Sim without depending on the Tancho v2 project.

### 1. Install the training environment

```bash
cd Tancho_v3_simready
chmod +x scripts/setup_tancho_v3_isaac_lab_env.sh
./scripts/setup_tancho_v3_isaac_lab_env.sh
conda activate env_isaaclab
python -m pip install -e source/tancho_v3_lab
```

### 2. Validate the task and training environment

```bash
python scripts/list_envs.py
python -u scripts/zero_agent.py \
  --task=TanchoV3-Flat-v0 --num_envs=8 --headless \
  --max_steps=200 --fail_mean_episode_steps=10 --fail_non_timeout_ratio=1.01
```

`scripts/list_envs.py` should list `TanchoV3-Flat-v0` and `TanchoV3-Rough-v0`, and zero-agent should finish with `ZERO_AGENT_GATE status=passed`. Rough inherits the Flat actions, observations, commands, terminations, events, curriculum, and simulation settings while using separate Rough scene, terrain, and reward configs.

Zero-agent is a low-cost preflight check before full training. It does not train a policy. It runs zero actions for 200 control steps in 8 environments to verify that Isaac Sim, CUDA, the URDF, action and observation shapes, rewards, terminations, and resets all execute successfully. Passing means the environment is ready to start training; it does not mean the robot has learned to stand.

### 3. Flat And Rough Configuration Structure

Environment files are located under:

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/
```

| File | Responsibility |
|---|---|
| `tancho_v3_env_cfg.py` | Robot, sensors, and shared MDP configuration |
| `flat_env_cfg.py` | Flat terrain, rewards, scene, and environment |
| `rough_env_cfg.py` | Rough terrain, rewards, scene, and environment |
| `custom_rewards.py` | Custom reward functions shared by Flat and Rough |
| `__init__.py` | Gym registration for `TanchoV3-Flat-v0` and `TanchoV3-Rough-v0` |

Inheritance:

```text
Shared MDP
├── Flat: terrain + rewards + scene
└── Rough: terrain + rewards + scene
```

Flat remains the minimum standing baseline. Rough keeps the same observation and action ordering while replacing the terrain and providing a separate reward config, so Rough terrain and reward changes do not require changes to Flat.

The scene creates an `ImuCfg` sensor 30 mm below the body center. The policy observations `base_lin_vel`, `base_ang_vel`, and `projected_gravity` currently still use articulation root state rather than reading the IMU sensor directly.

### 4. Start a fresh training run

```bash
python -u scripts/rsl_rl/train.py \
  --task=TanchoV3-Flat-v0 --num_envs=4096 --headless \
  --max_iterations=1500
```

For rough training, change the task to `TanchoV3-Rough-v0`.

- `--task`: selects the training task
- `--num_envs=4096`: number of environments running in parallel
- `--headless`: disables Isaac Sim window rendering
- `--max_iterations=1500`: maximum number of iterations

This starts a fresh run. Do not add `--resume`, `--load_run`, or `--checkpoint`.

### 5. TensorBoard and Play

```bash
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006

python scripts/rsl_rl/play.py \
  --task=TanchoV3-Flat-v0 --num_envs=1
```

Play loads the newest model from the newest run by default.

To select a model, append this option to the Play command:

```text
--checkpoint=logs/rsl_rl/tancho_v3/<run_timestamp>/model_x.pt
```

## AI Agent Integration Entrypoint (Optional)

`scripts/agent_entrypoint.sh`

```bash
bash scripts/agent_entrypoint.sh list-envs
bash scripts/agent_entrypoint.sh zero-agent
bash scripts/agent_entrypoint.sh train
bash scripts/agent_entrypoint.sh tensorboard
bash scripts/agent_entrypoint.sh play
bash scripts/agent_entrypoint.sh play logs/rsl_rl/tancho_v3/<run_timestamp>/model_x.pt
```

### Diagnostics and MCP

Engineering diagnostics are located under `scripts/diagnostics/`:

```bash
python scripts/diagnostics/pose_scan.py --task=TanchoV3-Flat-v0 --max_steps=600 --headless
python scripts/diagnostics/wheel_pd_scan.py --task=TanchoV3-Flat-v0 --max_steps=600 --headless
python scripts/diagnostics/wheel_pd_scan.py --task=TanchoV3-Flat-v0 --max_steps=600 --constant_scan --headless
```

`tools/isaaclab_mcp/` is an optional, restricted AI-agent control interface for zero-agent validation, both diagnostics, training, and Play. It does not accept arbitrary shell commands. Diagnostic inputs are limited to `max_steps` (1-10000) and the wheel-only `constant_scan` option.

## Validated Training Environment

This project has been validated with:

```text
Python 3.11
Isaac Sim 5.1.0
Isaac Lab 2.3.2
Torch 2.7.0+cu128
Torchvision 0.22.0+cu128
rsl_rl 5.0.1
```


## Training Outputs And Checkpoints

Training outputs are written under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

Useful commands:

```bash
find logs -name "*.pt"
find logs -name "*.pt" | sort | tail -n 1
find logs -name "policy.onnx"
```

Training saves `model_final.pt` and also attempts to export `exported/policies/policy.onnx`.

## Manually Resume Training

The command below is only for manually continuing an old run. Never load an old checkpoint after changing the observation or action dimensions.

```bash
python scripts/rsl_rl/train.py --task TanchoV3-Flat-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

## Export ONNX From A Checkpoint

```bash
python scripts/rsl_rl/play.py --task TanchoV3-Flat-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

## Modify Rewards

Flat terrain and rewards are defined in:

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/flat_env_cfg.py
```

Rough terrain and rewards are defined in:

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/rough_env_cfg.py
```

`RoughRewardsCfg` is located in `rough_env_cfg.py` and currently reuses the Flat weights. Redeclare a term there to change it only for Rough.

Add new reward calculation functions to:

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/custom_rewards.py
```

Then create a `RewardTerm` in `flat_env_cfg.py` or `rough_env_cfg.py` that references the function. Environment files define terms and weights; `custom_rewards.py` contains calculations only.

## Modify Terrain

The Flat terrain is defined in `flat_env_cfg.py`; Rough generator settings are defined in `rough_env_cfg.py`. Rough can independently change:

- `num_rows` and `num_cols`
- `difficulty_range`
- proportions of random rough terrain, slopes, and boxes
- height, slope, and grid dimensions

Run the 8-environment zero-agent check for both Flat and Rough after making changes.
