# Tancho

本專案包含 Tancho 機構的 CAD、URDF，以及 Isaac Lab / Isaac Sim 可訓練機器人導出工程。

## 目錄說明

- `Print.3mf`
  - 機構的 3D 列印配置檔案。
- `Tancho.step`
  - 全尺寸 Tancho CAD 模型。
- `Tancho_simplified.step`
  - 已簡化的 CAD 模型，目前未定義關節，適合作為初步幾何檢視或加工用。
- `Tencho_v1/`
  - 已設定好的 URDF 機器人模型，可直接載入 ROS / 物理引擎做初步測試。
  - 目前模型品質較低，屬於基本可用版本。
- `Tencho_v1_simready/`
  - 由 StackForce SimReady 匯出、可直接在 Isaac Lab / Isaac Sim 使用的訓練工程。
  - 包含訓練與部署腳本、Python 套件原始碼與資產檔。

## Tencho_v1_simready 內容

此子工程已包含以下功能：

- `scripts/`：環境列舉、零策略測試、訓練、播放等腳本。
- `source/stackforce_simready_tencho_v1_lab/`：Python 套件碼與 Isaac Lab 任務定義。
- `README.en.md`、`README.zh.md`：英文與中文使用說明。

## 快速上手

1. 進入 `Tencho_v1_simready` 目錄：

```bash
cd Tencho_v1_simready
```

2. 安裝本地套件：

```bash
python -m pip install -e source/stackforce_simready_tencho_v1_lab
```

3. 列出可用任務：

```bash
python scripts/list_envs.py
```

4. 執行零策略測試：

```bash
python scripts/zero_agent.py --task Template-TenchoV1-Direct-v0 --headless --num_envs 8
```

5. 開始訓練：

```bash
python scripts/rsl_rl/train.py --task Template-TenchoV1-Direct-v0 --headless --num_envs 64 --max_iterations 100
```

如果想要觀看 Isaac Sim 視窗，移除 `--headless`。

## 推薦環境

建議使用與匯出工程匹配的 Isaac Lab / Isaac Sim 環境：

- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2 / pip 2.3.2.post1
- Torch 2.7.0+cu128
- Torchvision 0.22.0+cu128
- LeggedGym-Ex 0.3.0

`Tencho_v1_simready/` 也含一鍵環境設定腳本：

```bash
chmod +x scripts/setup_stackforce_isaac_lab_sim_env.sh
./scripts/setup_stackforce_isaac_lab_sim_env.sh
```

可透過 `ENV_NAME` 參數更改環境名稱。

## 常用命令

- 繼續訓練：

```bash
python scripts/rsl_rl/train.py --task Template-TenchoV1-Direct-v0 --resume --load_run <run_dir_name> --checkpoint <model_x.pt>
```

- 播放訓練策略：

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --num_envs 1 --disable_resets
```

- 匯出 ONNX：

```bash
python scripts/rsl_rl/play.py --task Template-TenchoV1-Direct-v0 --checkpoint logs/rsl_rl/<experiment_name>/<timestamp>/model_final.pt --num_envs 1 --disable_resets --export_onnx --num_steps 1
```

## 自定義 Reward

可修改以下檔案來加入自定義 reward：

- `source/stackforce_simready_tencho_v1_lab/stackforce_simready_tencho_v1_lab/tasks/direct/tencho_v1/custom_rewards.py`
- `source/stackforce_simready_tencho_v1_lab/stackforce_simready_tencho_v1_lab/tasks/direct/tencho_v1/tencho_v1_env_cfg.py`

將 `"custom_reward": 0.0` 改為非零值，例如 `1.0`，即可啟用自訂獎勵項目。

## 備註

- 若只要查看模型幾何，可使用 `Tancho.step` 或 `Tancho_simplified.step`。
- 若要進行機器人仿真與訓練，請優先使用 `Tencho_v1_simready/`。
- `Tencho_v1/` 提供基本 URDF 支持，可作為 ROS / 物理引擎整合的起點。
