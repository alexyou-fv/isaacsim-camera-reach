
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.
Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import os

UR_CAMERA_CFG = ArticulationCfg(

# Where is the USD file for this robot?
spawn=sim_utils.UsdFileCfg(       
    usd_path=os.path.expanduser('~/isaacsim-assets/ur5e-custom-gripper.usd'), 
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
    ),
# What is its initial position of the robot, and its joints?
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": -0.380,
            "shoulder_lift_joint": -1.696,
            "elbow_joint": 1.991,
            "wrist_1_joint": -0.290,
            "wrist_2_joint": 1.193,
            "wrist_3_joint": 3.148,
            "gripper_left_joint": 0.0,
            "gripper_right_joint": 0.0,
        },
    ),
# What parts of the robot move, and how stiff / damped are they?
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=87.0,
            stiffness=100.0,
            damping=40.0,
        ),
        # "gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["finger_joint"],
        #     stiffness=280,
        #     damping=28
        # ),
    }
)
