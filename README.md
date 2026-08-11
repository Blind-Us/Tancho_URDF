# Tancho - 6-DOF Wheel-Legged Robot

Tancho 是一個 6 自由度（6-DOF）雙輪腿機器人專案。本專案包含 CAD 機構設計模型、3D 列印檔，以及基於 NVIDIA Isaac Lab 與 RSL_RL 進行強化學習（RL）訓練的仿真環境。

---

## 目錄結構 (Directory Structure)

Tancho/
├── Tancho_v2/              # V2 URDF 與 Mesh 模型資源檔
├── Tancho_v2_simready/    # Isaac Lab 一鍵部屬訓練
├── Print.3mf              # 3D 列印切片設定檔 (Bambu Studio)
├── Tancho.step            # 完整 STEP 模型
├── Tancho_simplified.step # 簡化版 STEP 模型 （用於轉換URDF)
└── README.md              # U r looking at

---

## 版本演進 (Version History)

### V1 - 首次導出 (Initial Export)
* 完成基本 3D CAD 模型建立與 URDF 首次導出。
* 包含各剛體（Links）與關節（Joints）的初始幾何資訊。

---

### V2 - 模型精確化與 RL 環境架設 (Current)
* 修復 V1 導出時 Mesh 不精確、質量慣量與幾何中心偏差問題。
* URDF 頂層加入 Dummy Root Link 並修正 X 軸旋轉與生成撞地。
* 採用 Isaac Lab `ManagerBasedRLEnv` 寫法，於 `tancho_v2_env_cfg.py` 配置完整的 MDP 框架：

---

## 快速開始 (Quick Start)

### 1. 啟動 PPO 強化學習訓練
進到 Tancho_v2_simready 目錄並執行訓練指令：

conda activate env_isaaclab
cd Tancho_v2_simready
python scripts/rsl_rl/train.py --task=Template-TanchoV2-Direct-v0 --num_envs=4096 --headless

---

### 2. 模型評估與視覺化展示 (Play)
載入訓練好的模型檔進行推論測試：
python scripts/rsl_rl/play.py --task=Template-TanchoV2-Direct-v0 --num_envs=16

---

### 3. 即時監控訓練曲線
開啟 TensorBoard 觀察 Reward 與 Loss 變化：

tensorboard --logdir=logs/rsl_rl/

---

### 模擬環境與MDP函數路徑
./Tancho/Tancho_v2_simready/source/stackforce_simready_tancho_v2_lab/stackforce_simready_tancho_v2_lab/tasks/direct/tancho_v2
