# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg, FrameTransformerCfg
import torch
import os

from . import mdp

##
# Pre-defined configs
##

from .ur_with_camera import UR_CAMERA_CFG


##
# Scene definition
##


cube_width = 0.06

@configclass
class ReachBaseSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # robot
    robot: ArticulationCfg = UR_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.expanduser('~/isaacsim-assets/seattlelabtable/table_instanceable.usd'),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Cube
    target_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=tuple([cube_width] * 3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),

        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, cube_width / 2)),
        
    )


@configclass
class ReachVisionSceneCfg(ReachBaseSceneCfg):

    # Cameras
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/topdowncamera",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.50, 0.0, 2.0), convention="ros"),
    )

    hand_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur5e/wrist_3_link/sensors/down_cam",
        update_period=0.1,
        height=128,
        width=128,
        data_types=['rgb'],
        update_latest_camera_pose=True,
        spawn=None,
    )

    # camera_pose_tracker = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/ee_link"),
    #         # FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/hand_camera")
    #     ],
    #     debug_vis=False
    # )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], 
        scale=1, 
        use_default_offset=True, 
        debug_vis=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        gt_cube_pos = ObsTerm(func=mdp.cube_pos_from_robot, params={'cube_cfg': SceneEntityCfg('target_cube'), 'robot_cfg': SceneEntityCfg('robot')})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    class InternalObsCfg(ObsGroup):
        """
        These observations are ignored but used to compute the state
        """
        ee_stable = ObsTerm(func=mdp.check_stable, params={'robot_cfg': SceneEntityCfg('robot')})
        cube_in_fingers = ObsTerm(func=mdp.check_cube_in_fingers, params={'robot_cfg': SceneEntityCfg('robot'), 'cube_cfg': SceneEntityCfg('target_cube')})

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    internal: InternalObsCfg = InternalObsCfg()


@configclass
class ObservationsVisionCfg:
    
    @configclass
    class PolicyCfg(ObsGroup):

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))        
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        top_camera = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"}
        )
        gt_cube_pos = ObsTerm(func=mdp.cube_pos_from_robot, params={'cube_cfg': SceneEntityCfg('target_cube'), 'robot_cfg': SceneEntityCfg('robot')})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    class InternalObsCfg(ObsGroup):
        """
        These observations are ignored but used to compute the state
        """
        ee_stable = ObsTerm(func=mdp.check_stable, params={'robot_cfg': SceneEntityCfg('robot')})
        cube_in_fingers = ObsTerm(func=mdp.check_cube_in_fingers, params={'robot_cfg': SceneEntityCfg('robot'), 'cube_cfg': SceneEntityCfg('target_cube')})

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    internal: InternalObsCfg = InternalObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_cube_location = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode='reset',
        params={
            'pose_range': {
                'x': (0.1, 0.30),
                'y': (-0.35, 0.35),
                # 'z': (0, 0.001),
            },
            'velocity_range': {},
            'asset_cfg': SceneEntityCfg('target_cube'),
        }

    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # We will penalize aliveness to encourage the robot to move to the target as quickly as possible
    alive = RewTerm(func=mdp.is_alive, weight=-0.2)
    
    # Want the EE to point down
    # Make sure the weight doesn't override the alive penalty
    point_down_reward = RewTerm(
        func=mdp.ee_pointing_direction,
        weight=0.2,
        params={'robot_cfg': SceneEntityCfg('robot'), 'ee_point_axis': 1}
    )

    # Closeness term
    close_reward = RewTerm(
        func=mdp.ee_is_close_to_target_cube,
        weight=1.0,
        params={'cube_cfg': SceneEntityCfg('target_cube'), 'robot_cfg': SceneEntityCfg('robot'), 'dist_threshold': 0.20}
    )

    in_fingers_reward = RewTerm(
        func=mdp.cube_in_between_reward,
        weight=2.0,
        params={'max_horiz_dist': 0.03, 'vert_dist_threshold': 0.10},
    )
    
    # Episode finish term
    finish_reward = RewTerm(
        func=mdp.is_stable_reward,
        weight=1.0,
        params={'cube_cfg': SceneEntityCfg('target_cube'), 'robot_cfg': SceneEntityCfg('robot'), 'reward_coeff': 5.0}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    stable = DoneTerm(
        func=mdp.is_stable,
    )


##
# Environment configuration
##


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     action_rate = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
#     )

#     joint_vel = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
#     )


@configclass
class ReachBaseEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene = ReachBaseSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3.0
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation

@configclass
class ReachVisionEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene = ReachVisionSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsVisionCfg = ObservationsVisionCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3.0
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation