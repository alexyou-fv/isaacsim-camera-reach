# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

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



def ee_is_close_to_target_cube(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                               dist_threshold=0.10, max_reward=1.0) -> torch.Tensor:
    
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]

    # ee_idx = robot.find_bodies(robot_cfg.body_names)[0][0]
    ee_idx = 7
    dist = get_target_cube_dist(robot, cube, ee_idx=ee_idx)

    return (1 - torch.clamp(dist / dist_threshold, min=0.0, max=1.0)) * max_reward


def goal_reached_terminate(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                           dist_threshold=0.01):
    
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    ee_idx = 7

    dists = get_target_cube_dist(robot, cube, ee_idx=ee_idx)
    return dists < dist_threshold



def goal_reached_terminate_reward(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                                  dist_threshold=0.01, reward_coeff=2.0) -> torch.Tensor:
    
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    ee_idx = 7

    dists = get_target_cube_dist(robot, cube, ee_idx=ee_idx)
    remaining_steps = env.max_episode_length - env.episode_length_buf
    reward = torch.zeros_like(dists)
    reached_goal = dists < dist_threshold
    reward[reached_goal] = remaining_steps[reached_goal] * reward_coeff
    
    return reward
    
    

