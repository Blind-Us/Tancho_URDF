# StackForce SimReady Isaac Lab Export

This project was generated from StackForce SimReady and is ready to train in Isaac Lab / Isaac Sim.

### 1. Install the training environment

```bash
cd Tancho_v2_simready
chmod +x scripts/setup_stackforce_isaac_lab_sim_env.sh
./scripts/setup_stackforce_isaac_lab_sim_env.sh
conda activate env_isaaclab
python -m pip install -e source/stackforce_simready_tancho_v2_lab
```

### 2. Validate the task and training environment

```bash
python scripts/list_envs.py
python -u scripts/zero_agent.py \
  --task=Template-TanchoV2-Direct-v0 --num_envs=8 --headless \
  --max_steps=200 --fail_mean_episode_steps=10 --fail_non_timeout_ratio=1.01
```

`scripts/list_envs.py` should list `Template-TanchoV2-Direct-v0`, and zero-agent should finish with `ZERO_AGENT_GATE status=passed`.

Zero-agent is a low-cost preflight check before full training. It does not train a policy. It runs zero actions for 200 control steps in 8 environments to verify that Isaac Sim, CUDA, the URDF, action and observation shapes, rewards, terminations, and resets all execute successfully. Passing means the environment is ready to start training; it does not mean the robot has learned to stand.

### 3. Start a fresh training run

```bash
python -u scripts/rsl_rl/train.py \
  --task=Template-TanchoV2-Direct-v0 --num_envs=4096 --headless \
  --max_iterations=1500
```

- `--task`: selects the training task
- `--num_envs=4096`: number of environments running in parallel
- `--headless`: disables Isaac Sim window rendering
- `--max_iterations=1500`: maximum number of iterations

This starts a fresh run. Do not add `--resume`, `--load_run`, or `--checkpoint`.

### 4. TensorBoard and Play

```bash
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006

python scripts/rsl_rl/play.py \
  --task=Template-TanchoV2-Direct-v0 --num_envs=1
```

Play loads the newest model from the newest run by default.

To select a model, append this option to the Play command:

```text
--checkpoint=logs/rsl_rl/tancho_v2/<run_timestamp>/model_x.pt
```

## AI Agent Integration Entrypoint (Optional)

`scripts/agent_entrypoint.sh`

```bash
bash scripts/agent_entrypoint.sh list-envs
bash scripts/agent_entrypoint.sh zero-agent
bash scripts/agent_entrypoint.sh train
bash scripts/agent_entrypoint.sh tensorboard
bash scripts/agent_entrypoint.sh play
bash scripts/agent_entrypoint.sh play logs/rsl_rl/tancho_v2/<run_timestamp>/model_x.pt
```

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
python scripts/rsl_rl/train.py --task Template-TanchoV2-Direct-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

## Export ONNX From A Checkpoint

```bash
python scripts/rsl_rl/play.py --task Template-TanchoV2-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

## Add A Custom Reward

Edit:

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/custom_rewards.py
```

Implement your reward terms in `compute_custom_reward_terms(env)`.
Then edit:

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/tancho_v2_env_cfg.py
```

Change:

```python
"custom_reward": 0.0
```

to a non-zero scale such as:

```python
"custom_reward": 1.0
```
