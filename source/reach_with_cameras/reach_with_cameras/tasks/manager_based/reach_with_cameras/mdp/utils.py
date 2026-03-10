import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def check_stable(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, hold_time=1.0, dist_threshold=0.01, ee_idx=7):

    

    cur_info = env.extras.get('ee_pos_queue')
    

    if cur_info is None:    
        n = max(int(hold_time / env.step_dt), 1)
        cur_info = torch.zeros((env.num_envs, n, 3), device=env.device)
        env.extras['ee_pos_queue'] = cur_info
    else:
        n = cur_info.shape[1]

    import ipdb
    ipdb.set_trace()

    ep_lens = env.episode_length_buf
    can_check = ep_lens >= n
    cur_pos = env.scene[robot_cfg.name].data.body_pos_w[:, ee_idx]

    final = torch.zeros((env.num_envs, ), device=env.device)
    # The logic in ep_lens % n is not correct right now, please verify
    final[can_check] = torch.norm(cur_pos[can_check] - cur_info[:, ep_lens % n][can_check], dim=1)

    cur_info[ep_lens % n] = cur_pos
    env.extras['ee_stable'] = final

    return final


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