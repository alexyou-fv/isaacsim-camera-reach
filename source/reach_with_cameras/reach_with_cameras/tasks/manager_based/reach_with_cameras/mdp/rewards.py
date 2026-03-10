# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def get_target_cube_dist(robot, cube, ee_idx: int) -> torch.Tensor:

    cube_pos = cube.data.root_pos_w
    ee_pos = robot.data.body_pos_w[:, ee_idx]

    return torch.norm(ee_pos[:, :2] - cube_pos[:, :2], dim=1)



def ee_pointing_direction(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ee_idx: int=7, ee_point_axis=2):

    reference_vector = torch.Tensor([0.0, 0.0, -1.0]).cuda()
    robot = env.scene[robot_cfg.name]
    quats = robot.data.body_pose_w[:, ee_idx][:, 3:7]
    ee_local_vec = torch.zeros((quats.shape[0], 3), device=quats.device)
    ee_local_vec[:, ee_point_axis] = 1.0
    return (torch.sum(quat_apply(quats, ee_local_vec) * reference_vector.repeat((quats.shape[0], 1)), axis=1) + 1) / 2
    

def ee_too_low_indicator(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, z_val: float, ee_idx: int = 7) -> torch.Tensor:
    
    robot = env.scene[robot_cfg.name]
    ee_pos = robot.data.body_pos_w[:, ee_idx]
    rewards = torch.zeros((ee_pos.shape[0], ), device=ee_pos.device)
    rewards[ee_pos[:, 2] < z_val] = 1.0

    return rewards

def ee_is_close_to_target_cube(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                               dist_threshold=0.10, max_reward=1.0) -> torch.Tensor:
    
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]

    # ee_idx = robot.find_bodies(robot_cfg.body_names)[0][0]
    ee_idx = 7
    dist = get_target_cube_dist(robot, cube, ee_idx=ee_idx)

    return (1 - torch.clamp(dist / dist_threshold, min=0.0, max=1.0)) * max_reward


def goal_reached_terminate(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                           dist_threshold=0.03):
    
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    ee_idx = 7

    dists = get_target_cube_dist(robot, cube, ee_idx=ee_idx)
    return dists < dist_threshold



def goal_reached_terminate_reward(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                                  dist_threshold=0.03, reward_coeff=2.0) -> torch.Tensor:
    

    # if not env.common_step_counter % 60:

    #     import cv2        
    #     data = env.scene.sensors['hand_camera'].data.output['rgb'][0].cpu().numpy()
    #     cv2.imwrite('test_downcam.png', data)

    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    ee_idx = 7

    dists = get_target_cube_dist(robot, cube, ee_idx=ee_idx)
    remaining_steps = env.max_episode_length - env.episode_length_buf
    reward = torch.zeros_like(dists)
    reached_goal = dists < dist_threshold
    reward[reached_goal] = remaining_steps[reached_goal] * reward_coeff
    
    return reward
    
    
