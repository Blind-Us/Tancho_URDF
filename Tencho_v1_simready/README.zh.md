# StackForce SimReady Isaac Lab 导出工程

这个工程由 StackForce SimReady 导出，可直接用于 Isaac Lab / Isaac Sim 训练。

### 直接可复制的训练命令

```bash
conda activate <你自己的IsaacLab环境名称>
cd <exported_project>
python -m pip install -e source/stackforce_simready_tencho_v1_lab
python scripts/list_envs.py
python scripts/zero_agent.py --task Template-TenchoV1-Direct-v0 --headless --num_envs 8
python scripts/rsl_rl/train.py --task Template-TenchoV1-Direct-v0 --headless --num_envs 64 --max_iterations 100
```

如果你想打开 Isaac Sim 窗口，把训练命令里的 `--headless` 去掉即可。

### 推荐 Isaac Lab / Isaac Sim 环境

本导出工程推荐使用下面这套已验证配置：

```text
Python 3.11
Isaac Sim 5.1.0
Isaac Lab v2.3.2 / pip 2.3.2.post1
Torch 2.7.0+cu128
Torchvision 0.22.0+cu128
LeggedGym-Ex 0.3.0 提供的 rsl_rl
```

导出包内已包含一键环境脚本：

```bash
chmod +x scripts/setup_stackforce_isaac_lab_sim_env.sh
./scripts/setup_stackforce_isaac_lab_sim_env.sh
```

脚本默认创建 `env_isaaclab`。如果你想改环境名：

```bash
ENV_NAME=my_isaaclab ./scripts/setup_stackforce_isaac_lab_sim_env.sh
```


### 训练输出和 checkpoint

训练输出在：

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

可用下面的命令直接查找：

```bash
find logs -name "*.pt"
find logs -name "*.pt" | sort | tail -n 1
find logs -name "policy.onnx"
```

Training saves `model_final.pt` and also attempts to export `exported/policies/policy.onnx`.

### 继续训练

```bash
python scripts/rsl_rl/train.py --task Template-TenchoV1-Direct-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

### 播放训练后的策略

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --num_envs 1 --disable_resets
```

如需加载指定 checkpoint：

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets
```

Regenerate ONNX from a checkpoint:

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

### 增加自定义 Reward

编辑：

```text
source/stackforce_simready_tencho_v1_lab/stackforce_simready_tencho_v1_lab/tasks/direct/tencho_v1/custom_rewards.py
```

在 `compute_custom_reward_terms(env)` 中返回你自己的 reward term。
然后再编辑：

```text
source/stackforce_simready_tencho_v1_lab/stackforce_simready_tencho_v1_lab/tasks/direct/tencho_v1/tencho_v1_env_cfg.py
```

把：

```python
"custom_reward": 0.0
```

改成非零，比如：

```python
"custom_reward": 1.0
```
