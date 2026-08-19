"""Tancho v2 自訂 reward / curriculum 函式。

注意：關節名稱以 URDF 為準 (joint_thigh_L / joint_calf_L / ...)，
不是 L_thigh_joint 這種舊命名。
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def nominal_state_thigh(env: ManagerBasedRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """懲罰左右腿 Thigh 關節的對稱偏差 |L_thigh - R_thigh|。"""
    robot: Articulation = env.scene[asset_cfg_name]
    l_idx, _ = robot.find_joints("joint_thigh_L")
    r_idx, _ = robot.find_joints("joint_thigh_R")
    diff = robot.data.joint_pos[:, l_idx[0]] - robot.data.joint_pos[:, r_idx[0]]
    return torch.abs(diff)


def nominal_state_calf(env: ManagerBasedRLEnv, asset_cfg_name: str = "robot") -> torch.Tensor:
    """懲罰左右腿 Calf 關節的對稱偏差 |L_calf - R_calf|。"""
    robot: Articulation = env.scene[asset_cfg_name]
    l_idx, _ = robot.find_joints("joint_calf_L")
    r_idx, _ = robot.find_joints("joint_calf_R")
    diff = robot.data.joint_pos[:, l_idx[0]] - robot.data.joint_pos[:, r_idx[0]]
    return torch.abs(diff)


def wheel_ground_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """輪子貼地獎勵：兩輪接觸力 > threshold 時給 1，否則 0。

    輪腿機器人需要輪子保持接觸地面才能用摩擦平衡。
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # 找 wheel body 的 index
    body_ids, _ = contact_sensor.find_bodies(".*wheel.*")
    forces = contact_sensor.data.net_forces_w[:, body_ids, :]
    contact = (torch.norm(forces, dim=-1) > threshold).float()
    # 兩輪都接觸 => 1
    return torch.prod(contact, dim=1)


def action_smoothness_2nd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """二階動作平滑度懲罰 (a_t - 2*a_{t-1} + a_{t-2})^2，針對腿部 4 關節。"""
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    if hasattr(env.action_manager, "prev_prev_action"):
        prev_prev_actions = env.action_manager.prev_prev_action
        acc = actions[:, :4] - 2 * prev_actions[:, :4] + prev_prev_actions[:, :4]
    else:
        acc = actions[:, :4] - prev_actions[:, :4]
    return torch.sum(torch.square(acc), dim=1)


def wheel_under_com_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_wheel_body: str,
    right_wheel_body: str,
    error_scale: float,
) -> torch.Tensor:
    """懲罰整機 COM 在水平面上偏離左右輪軸線的正規化距離。"""

    robot: Articulation = env.scene[asset_cfg.name]

    # 真正的各 rigid body COM 世界座標：[num_envs, num_bodies, 3]
    body_com_pos_w = robot.data.body_com_pos_w

    # PhysX 的 default_mass 在目前 Isaac Lab 版本保留在 CPU，而 body pose 在 CUDA。
    # 質量隨機化目前是 no-op，所以首次轉移後快取在 env，避免每步 CPU→GPU copy。
    mass_cache_name = "_tancho_default_body_mass"
    body_mass = getattr(env, mass_cache_name, None)
    if body_mass is None or body_mass.device != body_com_pos_w.device:
        body_mass = robot.data.default_mass.to(
            device=body_com_pos_w.device,
            dtype=body_com_pos_w.dtype,
        )
        setattr(env, mass_cache_name, body_mass)

    # 整機質心
    total_mass = body_mass.sum(dim=1, keepdim=True)
    robot_com_w = (
        body_com_pos_w * body_mass.unsqueeze(-1)
    ).sum(dim=1) / total_mass

    left_ids, _ = robot.find_bodies(left_wheel_body)
    right_ids, _ = robot.find_bodies(right_wheel_body)

    left_pos_w = robot.data.body_pos_w[:, left_ids[0], :]
    right_pos_w = robot.data.body_pos_w[:, right_ids[0], :]

    # 左右輪中心與輪軸方向，只取水平 XY
    wheel_mid_xy = 0.5 * (left_pos_w[:, :2] + right_pos_w[:, :2])
    axle_xy = right_pos_w[:, :2] - left_pos_w[:, :2]
    axle_xy = axle_xy / torch.clamp(
        torch.linalg.vector_norm(axle_xy, dim=1, keepdim=True),
        min=1.0e-6,
    )

    # COM 相對於輪軸中心的水平偏移
    delta_xy = robot_com_w[:, :2] - wheel_mid_xy

    # 移除沿輪軸方向的分量，只保留前後傾倒方向
    along_axle = torch.sum(delta_xy * axle_xy, dim=1, keepdim=True)
    perpendicular_error = delta_xy - along_axle * axle_xy

    # 用可調的物理容許誤差正規化，避免 m^2 數值過小而被其他 reward 蓋過。
    return torch.sum(torch.square(perpendicular_error), dim=1) / (error_scale**2)


# ---------------------------------------------------------------------------
# Curriculum modify_fn (不能用 lambda，Isaac Lab config 需要可序列化)
# ---------------------------------------------------------------------------

def curriculum_enable_velocity(env, env_ids, old_value, num_steps: int, target: float):
    """達到 num_steps 後，將 rel_standing_envs 從 1.0 降到 target。"""
    from isaaclab.envs.mdp.curriculums import modify_env_param
    if env.common_step_counter > num_steps:
        return target
    return modify_env_param.NO_CHANGE


def curriculum_enable_push(env, env_ids, old_value, num_steps: int, velocity_range: dict):
    """達到 num_steps 後，開啟推力干擾速度範圍。"""
    from isaaclab.envs.mdp.curriculums import modify_env_param
    if env.common_step_counter > num_steps:
        return velocity_range
    return modify_env_param.NO_CHANGE

def target_forward_pitch_l2(
    env: ManagerBasedRLEnv,
    target_gravity_x: float = 0.087,
    asset_cfg_name: str = "robot",
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg_name]
    gravity_x = robot.data.projected_gravity_b[:, 0]
    return torch.square(gravity_x - target_gravity_x)
