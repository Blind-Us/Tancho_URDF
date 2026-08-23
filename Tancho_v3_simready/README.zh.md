# Tancho V3 Isaac Lab 訓練工程

本工程使用獨立的 `tancho_v3_lab` package，可直接用於 Isaac Lab / Isaac Sim 訓練，不依賴 Tancho v2 工程。


### 1. 安裝訓練環境

```bash
cd Tancho_v3_simready
chmod +x scripts/setup_tancho_v3_isaac_lab_env.sh
./scripts/setup_tancho_v3_isaac_lab_env.sh
conda activate env_isaaclab
python -m pip install -e source/tancho_v3_lab
```

### 2. 驗證 task 與訓練環境

```bash
python scripts/list_envs.py
python -u scripts/zero_agent.py \
  --task=TanchoV3-Flat-v0 --num_envs=8 --headless \
  --max_steps=200 --fail_mean_episode_steps=10 --fail_non_timeout_ratio=1.01
```

`scripts/list_envs.py` 應列出 `TanchoV3-Flat-v0` 與 `TanchoV3-Rough-v0`，zero-agent 結束時應顯示 `ZERO_AGENT_GATE status=passed`。Rough 繼承 Flat 的 Action、Observation、Command、Termination、Event、Curriculum 與模擬設定，但使用獨立的 Rough Scene、地形與 Reward config。

Zero-agent 是正式訓練前的低成本行前檢查。它不會訓練 policy，而是在 8 個環境中以 zero action 執行 200 control steps，確認 Isaac Sim、CUDA、URDF、action/observation shape、reward、termination 與 reset 能夠完整運作。通過只代表環境可開始訓練，不代表機器人已學會站立。

### 3. Flat 與 Rough 配置架構

環境檔案位於：

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/
```

| 檔案 | 用途 |
|---|---|
| `tancho_v3_env_cfg.py` | 機器人、感測器與共用 MDP 配置 |
| `flat_env_cfg.py` | Flat terrain、Reward、Scene 與環境配置 |
| `rough_env_cfg.py` | Rough terrain、Reward、Scene 與環境配置 |
| `custom_rewards.py` | Flat/Rough 可共用的自訂 Reward 計算函數 |
| `__init__.py` | `TanchoV3-Flat-v0` 與 `TanchoV3-Rough-v0` 的 Gym 註冊 |

繼承關係：

```text
共用 MDP
├── Flat：terrain + reward + scene
└── Rough：terrain + reward + scene
```

Flat 保持最低站立基底。Rough 重用相同的 Observation 與 Action ordering，只替換地形並提供獨立 Reward config，因此調整 Rough 地形或權重時不必修改 Flat。

目前場景有建立 `ImuCfg`，位置為機身中心正下方 30 mm；Policy Observation 的 `base_lin_vel`、`base_ang_vel` 與 `projected_gravity` 仍來自 articulation root state，尚未改為直接讀取 IMU sensor。

### 4. 開始全新訓練

```bash
python -u scripts/rsl_rl/train.py \
  --task=TanchoV3-Flat-v0 --num_envs=4096 --headless \
  --max_iterations=1500
```

rough 訓練只需將 task 改為 `TanchoV3-Rough-v0`。

- `--task`：選擇訓練任務
- `--num_envs=4096`：同時並行環境數量
- `--headless`：不開啟 Isaac Sim 視窗渲染
- `--max_iterations=1500`：最多 iterations 次數

這是全新訓練，不要加入 `--resume`、`--load_run` 或 `--checkpoint`。

### 5. TensorBoard 與 Play

```bash
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006

python scripts/rsl_rl/play.py \
  --task=TanchoV3-Flat-v0 --num_envs=1
```

Play 預設載入最新 run 的最新 model。

如要指定 model，在 Play 指令最後加上：

```text
--checkpoint=logs/rsl_rl/tancho_v3/<run_timestamp>/model_x.pt
```

## AI Agent 接入點（選用）

`scripts/agent_entrypoint.sh`

```bash
bash scripts/agent_entrypoint.sh list-envs
bash scripts/agent_entrypoint.sh zero-agent
bash scripts/agent_entrypoint.sh train
bash scripts/agent_entrypoint.sh tensorboard
bash scripts/agent_entrypoint.sh play
bash scripts/agent_entrypoint.sh play logs/rsl_rl/tancho_v3/<run_timestamp>/model_x.pt
```

### 診斷工具與 MCP

工程診斷腳本位於 `scripts/diagnostics/`：

```bash
python scripts/diagnostics/pose_scan.py --task=TanchoV3-Flat-v0 --max_steps=600 --headless
python scripts/diagnostics/wheel_pd_scan.py --task=TanchoV3-Flat-v0 --max_steps=600 --headless
python scripts/diagnostics/wheel_pd_scan.py --task=TanchoV3-Flat-v0 --max_steps=600 --constant_scan --headless
```

`tools/isaaclab_mcp/` 為選用的 AI agent 安全控制介面，可啟動 zero-agent、上述兩種診斷、訓練與 Play。MCP 不接受任意 shell 命令；診斷參數只開放 `max_steps`（1–10000）以及 wheel scan 專用的 `constant_scan`。

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
python scripts/rsl_rl/train.py --task TanchoV3-Flat-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

## 從 Checkpoint 匯出 ONNX

```bash
python scripts/rsl_rl/play.py --task TanchoV3-Flat-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

## 修改 Reward

Flat terrain 與 Reward 在：

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/flat_env_cfg.py
```

Rough terrain 與 Reward 在：

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/rough_env_cfg.py
```

`RoughRewardsCfg` 位於 `rough_env_cfg.py`，目前沿用 Flat 權重；在該類別重新宣告同名 term 只會影響 Rough。

需要新的計算函數時，先加入：

```text
source/tancho_v3_lab/tancho_v3_lab/tasks/direct/tancho_v3/custom_rewards.py
```

再從 `flat_env_cfg.py` 或 `rough_env_cfg.py` 建立 `RewardTerm` 引用該函數。環境檔負責 term 與權重，`custom_rewards.py` 只負責計算。

## 修改地形

Flat 地形在 `flat_env_cfg.py`，Rough 的生成參數在 `rough_env_cfg.py`。Rough 可獨立調整：

- `num_rows`、`num_cols`
- `difficulty_range`
- 隨機粗糙地形、斜坡與方塊的比例
- 高度、坡度與網格尺寸

修改後先對 Flat 與 Rough 各執行一次 8-environment zero-agent 驗證。
