# Tancho 訓練規定

本文件適用於 Tancho V2 的 Isaac Lab / RSL_RL 強化學習訓練。

## 1. 規範依據

- 後續訓練以 Tancho 專案根目錄的 `./README.md` 為準。
- `Tancho_v2_simready/README.zh.md` 是 SimReady 匯出時附帶的說明，不作為訓練參數的主要規範。
- SimReady 匯出說明只用於專案下載後的快速部署與基本環境導入。

## 2. 專案下載後快速部署

進入 SimReady 訓練目錄，使用內附腳本建立 Isaac Lab / Isaac Sim 環境：

```bash
cd ./Tancho_v2_simready
chmod +x scripts/setup_stackforce_isaac_lab_sim_env.sh
./scripts/setup_stackforce_isaac_lab_sim_env.sh
```

腳本預設建立 `env_isaaclab` Conda 環境。若需要自訂環境名稱：

```bash
ENV_NAME=my_isaaclab ./scripts/setup_stackforce_isaac_lab_sim_env.sh
```

## 3. 基本環境導入

啟用環境、進入訓練目錄，並以 editable 模式安裝 Tancho 套件：

```bash
conda activate env_isaaclab
cd ./Tancho_v2_simready
python -m pip install -e source/stackforce_simready_tancho_v2_lab
```

首次部署或環境程式碼變更後，應先確認任務可以載入：

```bash
python scripts/list_envs.py
python scripts/zero_agent.py --task Template-TanchoV2-Direct-v0 --headless --num_envs 8
```

## 4. PPO 訓練規定

Tancho V2 的標準訓練任務為 `Template-TanchoV2-Direct-v0`，基準指令如下：

```bash
conda activate env_isaaclab
cd ./Tancho_v2_simready
python scripts/rsl_rl/train.py --task=Template-TanchoV2-Direct-v0 --num_envs=4096 --headless
```

- 正式批次訓練預設使用 `4096` 個環境及 `--headless`。
- 若需要 Isaac Sim 視窗進行觀察，可移除 `--headless`。
- 根目錄 README 沒有固定最大迭代次數；如需加入 `--max_iterations`，應依該次實驗需求設定並記錄。
- 修改環境參數、觀測值、動作空間、Reward 或其他訓練參數後，應建立新的訓練實驗，避免混淆不同設定的結果。

## 5. 模型評估與視覺化

訓練完成後，使用根目錄 README 規定的基準指令進行推論測試：

```bash
python scripts/rsl_rl/play.py --task=Template-TanchoV2-Direct-v0 --num_envs=16
```

## 6. 訓練監控

使用 TensorBoard 觀察 Reward 與 Loss：

```bash
tensorboard --logdir=logs/rsl_rl/
```

評估訓練狀態時，應同時檢查 Reward、Loss 及其變化趨勢，不應只以程序是否仍在執行作為判斷依據。

## 7. 環境與 MDP 修改位置

環境參數、Observations、Actions 與 Rewards 的主要程式碼位於：

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/
```

主要環境與 MDP 設定檔為：

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/tancho_v2_env_cfg.py
```

## 8. 自訂 Reward 掛載點

自訂 Reward 的實作掛載點為：

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/custom_rewards.py
```

在 `compute_custom_reward_terms(env)` 中實作並回傳自訂 reward term。

完成實作後，還必須在下列設定檔啟用權重：

```text
source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2/tancho_v2_env_cfg.py
```

將：

```python
"custom_reward": 0.0
```

改為所需的非零權重，例如：

```python
"custom_reward": 1.0
```

自訂 Reward 只有在函式已實作且設定檔中的權重非零時才會實際掛載到訓練流程。修改後應先執行任務載入與 zero-agent 測試，再開始新的 PPO 訓練。
