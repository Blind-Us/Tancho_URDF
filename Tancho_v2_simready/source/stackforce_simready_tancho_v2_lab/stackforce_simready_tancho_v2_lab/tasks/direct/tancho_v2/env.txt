import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

def nominal_state_thigh(env: ManagerBasedRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """懲罰左右腿 Thigh 關節的對稱偏差 (L_thigh - R_thigh)"""
    robot: Articulation = env.scene[asset_cfg_name]
    
    # 取得關節 Index (find_joints 回傳 (indices_tensor, names))
    l_idx, _ = robot.find_joints("L_thigh_joint")
    r_idx, _ = robot.find_joints("R_thigh_joint")
    
    diff = robot.data.joint_pos[:, l_idx[0]] - robot.data.joint_pos[:, r_idx[0]]
    return torch.abs(diff)


def nominal_state_calf(env: ManagerBasedRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """懲罰左右腿 Calf 關節的對稱偏差 (L_calf - R_calf)"""
    robot: Articulation = env.scene[asset_cfg_name]
    
    l_idx, _ = robot.find_joints("L_calf_joint")
    r_idx, _ = robot.find_joints("R_calf_joint")
    
    diff = robot.data.joint_pos[:, l_idx[0]] - robot.data.joint_pos[:, r_idx[0]]
    return torch.abs(diff)


def action_smoothness_2nd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """二階動作平滑度懲罰 (Penalize action acceleration)
    
    公式: (a_t - 2 * a_{t-1} + a_{t-2})^2
    針對腿部關節 actions[:, :4]
    """
    # 取得目前與歷史 Action (歷史深度需在 action_manager 支援)
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    
    # 若有保存歷史前兩幀，從 action_term 提取；若只有 prev_action，則改用一階差分或完整二階公式：
    if hasattr(env.action_manager, "prev_prev_action"):
        prev_prev_actions = env.action_manager.prev_prev_action
        acc = actions[:, :4] - 2 * prev_actions[:, :4] + prev_prev_actions[:, :4]
    else:
        # 當沒有 a_{t-2} 時，退回為變更率的一階懲罰 (Action Rate)，避免公式少一項導致錯誤
        acc = actions[:, :4] - prev_actions[:, :4]
        
    return torch.sum(torch.square(acc), dim=1)
