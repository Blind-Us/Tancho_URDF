from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass


ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "generated_robot"
URDF_PATH = ASSET_DIR / "urdf" / "generated_robot.urdf"


@configclass
class TenchoV1EnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 6
    observation_space = 30
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(URDF_PATH),
            fix_base=False,
            merge_fixed_joints=True,
            self_collision=False,
            replace_cylinders_with_capsules=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1),
            joint_pos={
        "joint_thigh_L": 0.0,
        "joint_calf_L": 0.0,
        "joint_wheel_L": 0.0,
        "joint_thigh_R": 0.0,
        "joint_calf_R": 0.0,
        "joint_wheel_R": 0.0,
    },
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_thigh_L", "joint_calf_L", "joint_wheel_L", "joint_thigh_R", "joint_calf_R", "joint_wheel_R"],
                stiffness={
        "joint_thigh_L": 20.0,
        "joint_calf_L": 20.0,
        "joint_wheel_L": 20.0,
        "joint_thigh_R": 20.0,
        "joint_calf_R": 20.0,
        "joint_wheel_R": 20.0,
    },
                damping={
        "joint_thigh_L": 0.5,
        "joint_calf_L": 0.5,
        "joint_wheel_L": 0.5,
        "joint_thigh_R": 0.5,
        "joint_calf_R": 0.5,
        "joint_wheel_R": 0.5,
    },
                effort_limit_sim={
        "joint_thigh_L": 10.0,
        "joint_calf_L": 10.0,
        "joint_wheel_L": 10.0,
        "joint_thigh_R": 10.0,
        "joint_calf_R": 10.0,
        "joint_wheel_R": 10.0,
    },
                velocity_limit_sim={
        "joint_thigh_L": 5.0,
        "joint_calf_L": 5.0,
        "joint_wheel_L": 5.0,
        "joint_thigh_R": 5.0,
        "joint_calf_R": 5.0,
        "joint_wheel_R": 5.0,
    },
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.02,
        track_air_time=True,
    )

    command_ranges = {
        "lin_vel_x": (-0.5, 0.5),
        "lin_vel_y": (-0.25, 0.25),
        "ang_vel_yaw": (-0.5, 0.5),
    }

    reward_scales = {
        "termination": 100.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.0,
        "torques": -0.00001,
        "dof_vel": 0.0,
        "dof_acc": -2.5e-7,
        "base_height": -1.0,
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": 0.0,
        "action_rate": -0.01,
        "stand_still": 0.0,
        "custom_reward": 0.0,
    }
    default_joint_angles = {
        "joint_thigh_L": 0.0,
        "joint_calf_L": 0.0,
        "joint_wheel_L": 0.0,
        "joint_thigh_R": 0.0,
        "joint_calf_R": 0.0,
        "joint_wheel_R": 0.0,
    }
    actuated_joint_names = ["joint_thigh_L", "joint_calf_L", "joint_wheel_L", "joint_thigh_R", "joint_calf_R", "joint_wheel_R"]
    foot_link_names = ["wheel_L", "wheel_R"]
    base_link_name = "base_link"
    auto_ground_spawn_height = 1
    auto_ground_clearance = 0.02
    auto_ground_complete = False
    base_height_target = 0.98
    action_clip = 1.0
    visual_disable_resets = False
    termination_grace_time_s = 2.0
    termination_contact_force_threshold = 20.0
    fallen_projected_gravity_z = -0.35
    fallen_termination_time_s = 0.5
    min_base_height = 0.441
    randomize_initial_episode_length = False
