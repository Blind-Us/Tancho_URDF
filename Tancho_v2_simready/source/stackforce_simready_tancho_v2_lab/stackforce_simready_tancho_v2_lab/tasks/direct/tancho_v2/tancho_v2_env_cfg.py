# tancho_v2_env_cfg.py
import math
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    TerminationTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewardTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "Tancho_v2"
URDF_PATH = str(ASSET_DIR / "urdf" / "Tancho_v2.urdf")


@configclass
class PTanchoV2SceneCfg(InteractiveSceneCfg):
    """場景設定：地形與 6 DOF 輪腿機器人"""
    # 地形 (Plane)
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 環境光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.85, 0.85, 0.85),
        ),
    )
    
    # 機器人 (Tancho 6 DOF 輪腿機器人)
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
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "joint_thigh_L": -0.3,   
                "joint_calf_L": 0.5,     
                "joint_wheel_L": 0.0,
                "joint_thigh_R": -0.3,   
                "joint_calf_R": 0.5,     
                "joint_wheel_R": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            # 腿部：位置控制 (4 個腿關節)
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["joint_thigh_L", "joint_calf_L", 
                                  "joint_thigh_R", "joint_calf_R"],
                stiffness=25.0,
                damping=0.5,
            ),
            # 輪子：速度控制 (2 個輪關節)
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_L", "joint_wheel_R"],
                stiffness=0.0,
                damping=0.3,
            ),
        },
    )
    # 觸擊感測器 (修改包含根節點以精準捕捉 Base 碰撞)
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )


@configclass
class ActionsCfg:
    """動作空間設定 (腿部 Pos, 輪子 Vel)"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"],
        scale=0.5,
        use_default_offset=True,
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["joint_wheel_L", "joint_wheel_R"],
        scale=5.0,
        use_default_offset=False,
    )


@configclass
class CommandsCfg:
    """指令生成器設定"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-3.14, 3.14),
            heading=(-3.14, 3.14),
        ),
    )


@configclass
class ObservationsCfg:
    """觀測空間設定"""
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 輸入的觀測資料"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """對應 6 DOF 的純內建 (MDP) 獎勵項目"""

    # --- Task Tracking ---
    tracking_lin_vel = RewardTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    tracking_ang_vel = RewardTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # --- Base Penalties ---
    base_height = RewardTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.2},
    )
    lin_vel_z = RewardTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewardTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewardTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # --- Nominal Joint Pos Penalties ---
    joint_deviation = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"]
            )
        },
    )

    # --- Joint & Action Regularization ---
    dof_vel = RewardTerm(func=mdp.joint_vel_l2, weight=-5e-5)
    dof_acc = RewardTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    torques = RewardTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    action_rate = RewardTerm(func=mdp.action_rate_l2, weight=-0.01)

    # --- Safety & Limits ---
    collision = RewardTerm(
        func=mdp.undesired_contacts,
        weight=-100.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh.*", ".*base.*"]),
        },
    )
    dof_pos_limits = RewardTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
    )
    dof_vel_limits = RewardTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9},  
    )


@configclass
class TerminationsCfg:
    """終止條件設定"""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*base.*", ".*thigh.*", ".*calf.*"]),
            "threshold": 1.0,
        },
    )


@configclass
class EventCfg:
    """Domain Randomization 與 重置 (Reset) 設定"""

    # --- 重置邏輯 (Reset Events: 觸發 Terminate 時執行) ---
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 0.6),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # --- 隨機化邏輯 (Domain Randomization) ---
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class PTanchoV2EnvCfg(ManagerBasedRLEnvCfg):
    """主環境 Config"""
    scene: PTanchoV2SceneCfg = PTanchoV2SceneCfg(num_envs=4096, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005  # Control freq = 50 Hz