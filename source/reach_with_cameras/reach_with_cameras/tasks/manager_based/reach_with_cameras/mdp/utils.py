import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def cube_pos_from_robot(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg, ee_idx: int = 7):

    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]

    cube_pos = cube.data.root_pos_w
    ee_pos = robot.data.body_pos_w[:, ee_idx]

    return cube_pos - ee_pos


# def reset_cube_position(
#     # This function has a bug where it resets all the objects to the world position!
#     env: ManagerBasedRLEnv, 
#     env_ids: torch.Tensor, 
#     x_range: tuple[float, float], 
#     y_range: tuple[float, float],
#     asset_cfg: SceneEntityCfg
# ):
#     """Randomize the XY position of the cube at reset."""
#     # Extract the cube asset from the scene
#     cube = env.scene[asset_cfg.name]
#     # Clone the default root state (pos, rot, vel) for the environments being reset
#     new_pose = cube.data.default_root_state[env_ids][:,:7].clone()
    
#     # Generate random XY offsets
#     random_x = torch.empty(len(env_ids), device=env.device).uniform_(*x_range)
#     random_y = torch.empty(len(env_ids), device=env.device).uniform_(*y_range)
    
#     # Apply offsets to the X and Y coordinates (indices 0 and 1)
#     new_pose[:, 0] += random_x
#     new_pose[:, 1] += random_y
    
#     # # Write the updated state back to the simulation
#     cube.write_root_pose_to_sim(new_pose, env_ids)