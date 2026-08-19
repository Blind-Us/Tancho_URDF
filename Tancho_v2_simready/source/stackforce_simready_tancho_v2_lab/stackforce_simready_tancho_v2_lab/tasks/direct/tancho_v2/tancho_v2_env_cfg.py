# tancho_v2_env_cfg.py
"""Tancho v2 — 6 DOF 輪腿式機器人 Isaac Lab MDP 環境 (Manager-Based)。

設計原則
--------
1. 目前以平地直立站穩為目標；保留速度追蹤與 curriculum term，以零權重或門檻停用。
2. 只有輪子可以接地；base/thigh/calf 接地會強懲罰並終止，避免趴地刷分。
3. 站姿預設 thigh=-0.50 rad、calf=+1.00 rad，base 目標高度 0.299 m。
4. Mesh 為 local frame (STL 原點 == 關節軸心)，URDF visual/collision origin 全為 0。
"""
from pathlib import Path
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewardTerm,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from . import custom_rewards as cr

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "Tancho_v2"
URDF_PATH = str(ASSET_DIR / "urdf" / "Tancho_v2.urdf")

# -- 站姿參數 ---------------------------------------------------------------
# 腿節長 0.15 m；此站姿讓 base 高度接近 0.34 m。
STAND_THIGH = -0.60 #-0.50
STAND_CALF = 1.10 #1.00
# Respawn 高度只負責安全放置機器人；reward 目標高度獨立調整。
RESPAWN_ROOT_HEIGHT = 0.30
BASE_HEIGHT_TARGET = 0.27
CURRICULUM_VEL_ON_STEPS = 1_000_000_000
CURRICULUM_PUSH_ON_STEPS = 1_000_000_000


@configclass
class PTanchoV2SceneCfg(InteractiveSceneCfg):
    """場景設定：平地 + 6 DOF 輪腿機器人。"""

    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_PATH,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=None,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, RESPAWN_ROOT_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint_thigh_L": STAND_THIGH,
                "joint_calf_L": STAND_CALF,
                "joint_wheel_L": 0.0,
                "joint_thigh_R": STAND_THIGH,
                "joint_calf_R": STAND_CALF,
                "joint_wheel_R": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            # 腿部 4 關節：位置控制 (implicit PD)
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["joint_thigh_L", "joint_calf_L",
                                  "joint_thigh_R", "joint_calf_R"],
                stiffness=25.0,
                damping=0.75,
                effort_limit_sim=40.0,
                velocity_limit_sim=1.0,
            ),
            # 輪子 2 關節：速度控制 (零剛度)
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_L", "joint_wheel_R"],
                stiffness=0.0,
                damping=0.3,
                effort_limit_sim=10.0,
                velocity_limit_sim=20.0,
            ),
        },
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class ActionsCfg:
    """動作空間：腿部位置 (scale 0.2) + 輪子速度 (scale 5.0)。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"],
        scale=0.25, #1.0
        use_default_offset=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_wheel_L", "joint_wheel_R"],
        scale=2.0, #5.0
        use_default_offset=False,
    )


@configclass
class CommandsCfg:
    """速度指令保留；站立階段由 rel_standing_envs=1.0 固定為零指令。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,     # 100% env 初始為站立 (command = 0)
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.02, 0.02),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.57, 1.57),
            heading=(-3.14, 3.14),
        ),
    )


@configclass
class ObservationsCfg:
    """觀測空間。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)  
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """保留原有 MDP terms；站立階段不需要的項目使用零權重。"""

    alive = RewardTerm(func=mdp.is_alive, weight=1.0) #0.5
    termination_penalty = RewardTerm(func=mdp.is_terminated, weight=-200.0)

    # 保留原有自訂 term，但本階段只使用 Isaac Lab 內建 MDP reward。
    wheel_contact = RewardTerm(
        func=cr.wheel_ground_contact,
        weight=0.5, #0.0
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    )

    # 原有 tracking terms 保留，站立階段停用。
    tracking_lin_vel = RewardTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    tracking_ang_vel = RewardTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # 直立站穩主要使用既有 Isaac Lab MDP penalties。
    base_height = RewardTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": BASE_HEIGHT_TARGET},
    )
    forward_pitch = RewardTerm(
        func=cr.target_forward_pitch_l2,
        weight=-0.5,
        params={"target_gravity_x": 0.10,}  #反方向迅速傾倒便調負
    )
    flat_orientation = RewardTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    lin_vel_z = RewardTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewardTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    wheel_under_com = RewardTerm(
        func=cr.wheel_under_com_l2,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "left_wheel_body": "wheel_L",
            "right_wheel_body": "wheel_R",
            "error_scale": 0.05,
        },
    )

    # --- 關節偏離站姿 (只限腿 4 關節) ---
    joint_deviation = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"],
            )
        },
    )
    # --- 平滑與能量 ---
    dof_vel = RewardTerm(func=mdp.joint_vel_l2, weight=-5e-5)
    dof_acc = RewardTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    torques = RewardTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    action_rate = RewardTerm(func=mdp.action_rate_l2, weight=-0.01)

    collision = RewardTerm(
        func=mdp.undesired_contacts,
        weight=-10.0, #-50.0
        params={
            "threshold": 2.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link_root", ".*thigh.*", ".*calf.*"]
            ),
        },
    )
    dof_pos_limits = RewardTerm(func=mdp.joint_pos_limits, weight=-10.0)
    dof_vel_limits = RewardTerm(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9})


@configclass
class TerminationsCfg:
    """終止條件。"""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # 只有 base 碰地才終止；thigh/calf 碰撞保留在 collision reward 處理。
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link_root"]),
            "threshold": 10.0,
        },
    )
    # 保留原有 term；診斷階段以 pi 設為 no-op，不因姿態提前終止。
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi},
    )


@configclass
class EventCfg:
    """重置與 Domain Randomization。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)}, # "y": (-0.0, 0.1), "yaw": (-3.14, 3.14)
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 圍繞站姿小擾動 (default_joint_pos 來自 init_state)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 質量隨機化 (base 才 1.26 kg，範圍收斂)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link_root"]),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.8, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 原有推力 term 保留；速度範圍為零，因此站立階段不施加推力。
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class CurriculumCfg:
    """保留原有 curriculum；門檻高於本階段訓練長度，因此不會啟動。"""

    enable_velocity_commands = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "command_manager.cfg.base_velocity.rel_standing_envs",
            "modify_fn": cr.curriculum_enable_velocity,
            "modify_params": {"num_steps": CURRICULUM_VEL_ON_STEPS, "target": 0.5},
        },
    )
    enable_push = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "event_manager.cfg.push_robot.params.velocity_range",
            "modify_fn": cr.curriculum_enable_push,
            "modify_params": {
                "num_steps": CURRICULUM_PUSH_ON_STEPS,
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            },
        },
    )


@configclass
class PTanchoV2EnvCfg(ManagerBasedRLEnvCfg):
    """主環境 Config。"""

    scene: PTanchoV2SceneCfg = PTanchoV2SceneCfg(num_envs=4096, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005  # 200 Hz physics, 50 Hz control
