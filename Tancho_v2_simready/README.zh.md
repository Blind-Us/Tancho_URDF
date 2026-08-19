# StackForce SimReady Isaac Lab 匯出工程

本工程由 StackForce SimReady 匯出，可直接用於 Isaac Lab / Isaac Sim 訓練。


### 1. 安裝訓練環境

```bash
cd Tancho_v2_simready
chmod +x scripts/setup_stackforce_isaac_lab_sim_env.sh
./scripts/setup_stackforce_isaac_lab_sim_env.sh
conda activate env_isaaclab
python -m pip install -e source/stackforce_simready_tancho_v2_lab
```

### 2. 驗證 task 與訓練環境

```bash
python scripts/list_envs.py
python -u scripts/zero_agent.py \
  --task=Template-TanchoV2-Direct-v0 --num_envs=8 --headless \
  --max_steps=200 --fail_mean_episode_steps=10 --fail_non_timeout_ratio=1.01
```

`scripts/list_envs.py` 應列出 `Template-TanchoV2-Direct-v0`，zero-agent 結束時應顯示 `ZERO_AGENT_GATE status=passed`。

Zero-agent 是正式訓練前的低成本行前檢查。它不會訓練 policy，而是在 8 個環境中以 zero action 執行 200 control steps，確認 Isaac Sim、CUDA、URDF、action/observation shape、reward、termination 與 reset 能夠完整運作。通過只代表環境可開始訓練，不代表機器人已學會站立。

### 3. 開始全新訓練

```bash
python -u scripts/rsl_rl/train.py \
  --task=Template-TanchoV2-Direct-v0 --num_envs=4096 --headless \
  --max_iterations=1500
```

- `--task`：選擇訓練任務
- `--num_envs=4096`：同時並行環境數量
- `--headless`：不開啟 Isaac Sim 視窗渲染
- `--max_iterations=1500`：最多 iterations 次數

這是全新訓練，不要加入 `--resume`、`--load_run` 或 `--checkpoint`。

### 4. TensorBoard 與 Play

```bash
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006

python scripts/rsl_rl/play.py \
  --task=Template-TanchoV2-Direct-v0 --num_envs=1
```

Play 預設載入最新 run 的最新 model。

如要指定 model，在 Play 指令最後加上：

```text
--checkpoint=logs/rsl_rl/tancho_v2/<run_timestamp>/model_x.pt
```

## AI Agent 接入點（選用）

`scripts/agent_entrypoint.sh`

```bash
bash scripts/agent_entrypoint.sh list-envs
bash scripts/agent_entrypoint.sh zero-agent
bash scripts/agent_entrypoint.sh train
bash scripts/agent_entrypoint.sh tensorboard
bash scripts/agent_entrypoint.sh play
bash scripts/agent_entrypoint.sh play logs/rsl_rl/tancho_v2/<run_timestamp>/model_x.pt
```

## 已驗證的訓練環境

本工程已驗證的訓練組合：

```text
Python 3.11
Isaac Sim 5.1.0
Isaac Lab 2.3.2
Torch 2.7.0+cu128
Torchvision 0.22.0+cu128
rsl_rl 5.0.1
```

## 訓練輸出和 checkpoint

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

## 手動繼續訓練

下列指令僅用於手動延續舊 run。若 observation 或 action 維度有變更，不可載入舊 checkpoint。

```bash
python scripts/rsl_rl/train.py --task Template-TanchoV2-Direct-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

## 從 Checkpoint 匯出 ONNX

```bash
python scripts/rsl_rl/play.py --task Template-TanchoV2-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

## 新增自訂 Reward

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
