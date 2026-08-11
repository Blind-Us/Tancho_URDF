# StackForce SimReady Isaac Lab 匯出工程

本工程由 StackForce SimReady 匯出，可直接用於 Isaac Lab / Isaac Sim 訓練。

### 直接可複製的訓練指令

```bash
conda activate env_isaaclab
cd /media/azul/861896C11896B023/Tancho/Tancho_v2_simready
python -m pip install -e source/stackforce_simready_tancho_v2_lab
python scripts/list_envs.py
python scripts/zero_agent.py --task Template-TanchoV2-Direct-v0 --headless --num_envs 8
python scripts/rsl_rl/train.py --task Template-TanchoV2-Direct-v0 --headless --num_envs 4096 --max_iterations 1000
```

如果你想開啟 Isaac Sim 視窗，將訓練指令中的 `--headless` 拿掉即可。

### 推薦 Isaac Lab / Isaac Sim 環境

本匯出工程推薦使用下面這套已驗證配置：

```text
Python 3.11
Isaac Sim 5.1.0
Isaac Lab v2.3.2 / pip 2.3.2.post1
Torch 2.7.0+cu128
Torchvision 0.22.0+cu128
LeggedGym-Ex 0.3.0 提供的 rsl_rl
```

匯出包內已包含一鍵環境腳本：

```bash
chmod +x scripts/setup_stackforce_isaac_lab_sim_env.sh
./scripts/setup_stackforce_isaac_lab_sim_env.sh
```

腳本預設建立 `env_isaaclab`。如果你想改環境名稱：

```bash
ENV_NAME=my_isaaclab ./scripts/setup_stackforce_isaac_lab_sim_env.sh
```

### 訓練輸出和 checkpoint

訓練輸出部位於：

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

可以使用下面的指令直接尋找：

```bash
find logs -name "*.pt"
find logs -name "*.pt" | sort | tail -n 1
find logs -name "policy.onnx"
```

訓練會儲存 `model_final.pt`，且系統還會嘗試匯出 `exported/policies/policy.onnx`。

### 繼續訓練

```bash
python scripts/rsl_rl/train.py --task Template-TanchoV2-Direct-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

### 播放訓練後的策略

```bash
python scripts/rsl_rl/play.py --task Template-TanchoV2-Direct-v0 --num_envs 1 --disable_resets
```

如需載入指定 checkpoint：

```bash
python scripts/rsl_rl/play.py --task Template-TanchoV2-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets
```

從 Checkpoint 重新生成 ONNX：

```bash
python scripts/rsl_rl/play.py --task Template-TanchoV2-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

### 新增自訂 Reward

編輯：

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/custom_rewards.py
```

在 `compute_custom_reward_terms(env)` 中傳回你自己的 reward term。
然後再編輯：

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/tancho_v2_env_cfg.py
```

將：

```python
"custom_reward": 0.0
```

改成非零數值，例如：

```python
"custom_reward": 1.0
```
