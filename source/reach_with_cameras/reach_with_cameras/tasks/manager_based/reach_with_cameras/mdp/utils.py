import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import transform_points, pose_inv, quat_apply_inverse


def transform_points_to_pose_frame(pose_tensor, pts_tensor):
    frame_pos = pose_tensor[:, 0:3]
    frame_quat = pose_tensor[:, 3:7]

    delta_pos_w = pts_tensor - frame_pos    # Points to the point in question
    pts_local_tensor = quat_apply_inverse(frame_quat, delta_pos_w)
    return pts_local_tensor


def check_stable(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, hold_time=1.0, dist_threshold=0.01, ee_idx=7):

    cur_info = env.extras.get('ee_pos_queue')
    env_idxs = torch.arange(env.num_envs, device=env.device)

    if cur_info is None:    
        n = max(int(hold_time / env.step_dt), 1)
        cur_info = torch.zeros((env.num_envs, n, 3), device=env.device)
        env.extras['ee_pos_queue'] = cur_info
    else:
        n = cur_info.shape[1]

    ep_lens = env.episode_length_buf
    can_check = ep_lens >= n
    cur_pos = env.scene[robot_cfg.name].data.body_pos_w[:, ee_idx]
    
    final = torch.zeros((env.num_envs, ), device=env.device)
    # The logic in ep_lens % n is not correct right now, please verify
    final[can_check] = (torch.norm(cur_pos[can_check] - cur_info[env_idxs, ep_lens % n][can_check], dim=1) < dist_threshold) * 1.0

    cur_info[env_idxs, ep_lens % n] = cur_pos
    env.extras['ee_stable'] = final

    return final.reshape((-1, 1))

def check_cube_in_fingers(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg, finger_indexes: tuple[int] = (8, 9)):

    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]

    cube_pos = cube.data.root_pos_w
    finger_1_pose = robot.data.body_pose_w[:, finger_indexes[0]]
    finger_2_pose = robot.data.body_pose_w[:, finger_indexes[1]]

    cube_pos_in_finger_1 = transform_points_to_pose_frame(finger_1_pose, cube_pos)
    cube_pos_in_finger_2 = transform_points_to_pose_frame(finger_2_pose, cube_pos)
    
    cube_is_between = (cube_pos_in_finger_1[:, 0] > 0) & (cube_pos_in_finger_2[:, 0] > 0)
    vertical_offset = torch.max(torch.abs(cube_pos_in_finger_1[:, 2]), torch.abs(cube_pos_in_finger_2[:, 2]))
    horizontal_offset = torch.max(torch.abs(cube_pos_in_finger_1[:, 1]), torch.abs(cube_pos_in_finger_2[:, 1]))

    info = {
        'between': cube_is_between,
        'vertical': vertical_offset,
        'horizontal': horizontal_offset
    }

    env.extras['cube_position'] = info
    return (cube_is_between * 1.0).reshape((-1, 1))


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