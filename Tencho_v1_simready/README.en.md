# StackForce SimReady Isaac Lab Export

This project was generated from StackForce SimReady and is ready to train in Isaac Lab / Isaac Sim.

### Copy-Paste Training Commands

```bash
conda activate <your_isaac_lab_env_name>
cd <exported_project>
python -m pip install -e source/stackforce_simready_tencho_v1_lab
python scripts/list_envs.py
python scripts/zero_agent.py --task Template-TenchoV1-Direct-v0 --headless --num_envs 8
python scripts/rsl_rl/train.py --task Template-TenchoV1-Direct-v0 --headless --num_envs 64 --max_iterations 100
```

If you want the Isaac Sim window to open, remove `--headless`.

### Recommended Isaac Lab / Isaac Sim Environment

This export is recommended with the validated stack below:

```text
Python 3.11
Isaac Sim 5.1.0
Isaac Lab v2.3.2 / pip 2.3.2.post1
Torch 2.7.0+cu128
Torchvision 0.22.0+cu128
LeggedGym-Ex 0.3.0 bundled rsl_rl
```

The export includes a one-click environment script:

```bash
chmod +x scripts/setup_stackforce_isaac_lab_sim_env.sh
./scripts/setup_stackforce_isaac_lab_sim_env.sh
```

The default environment name is `env_isaaclab`. Override it with:

```bash
ENV_NAME=my_isaaclab ./scripts/setup_stackforce_isaac_lab_sim_env.sh
```


### Training Outputs And Checkpoints

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

### Resume Training

```bash
python scripts/rsl_rl/train.py --task Template-TenchoV1-Direct-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

### Play A Trained Policy

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --num_envs 1 --disable_resets
```

To load a specific checkpoint:

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets
```

Regenerate ONNX from a checkpoint:

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

### Add A Custom Reward

Edit:

```text
source/stackforce_simready_tencho_v1_lab/stackforce_simready_tencho_v1_lab/tasks/direct/tencho_v1/custom_rewards.py
```

Implement your reward terms in `compute_custom_reward_terms(env)`.
Then edit:

```text
source/stackforce_simready_tencho_v1_lab/stackforce_simready_tencho_v1_lab/tasks/direct/tencho_v1/tencho_v1_env_cfg.py
```

Change:

```python
"custom_reward": 0.0
```

to a non-zero scale such as:

```python
"custom_reward": 1.0
```

Notes:
- The StackForce `trimesh` option is mapped to Isaac Lab rough terrain generation.
- Assets stay embedded in the python package under `source/stackforce_simready_tencho_v1_lab/stackforce_simready_tencho_v1_lab/assets`.
