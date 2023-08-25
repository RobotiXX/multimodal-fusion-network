import numpy as np
import torch

def clip_velocities(gt_cmd_vel):        
    gt_cmd_vel[:,0] = np.clip(gt_cmd_vel[:,0], 0, 1.6)
    gt_cmd_vel[:,1] = np.clip(gt_cmd_vel[:,1], -0.55, 0.55)
    return gt_cmd_vel

def scale_min_max(gt_cmd_vel):
    gt_cmd_vel[:,0] = np.interp(gt_cmd_vel[:,0], (0, 1.6), (0, 10000))
    gt_cmd_vel[:,1] = np.interp(gt_cmd_vel[:,1], (-0.55, 0.55), (0, 20000))
    return gt_cmd_vel

def reverse_scale(gt_cmd_vel):
    gt_cmd_vel[:,0] = np.interp(gt_cmd_vel[:,0], (0, 10000), (0, 1.6))
    gt_cmd_vel[:,1] = np.interp(gt_cmd_vel[:,1], (0, 20000), (-0.55, 0.55))
    return gt_cmd_vel

def reverse_transform(cmd_vel_tensor):
    cmd_vel = cmd_vel_tensor.cpu().numpy()
    # print(cmd_vel.shape)
    cmd_scale_reversed = reverse_scale(cmd_vel)
    cmd_scale_reversed_tensor = torch.tensor(cmd_scale_reversed, dtype=torch.float32)
    return cmd_scale_reversed_tensor

def transform_cmd_vel(gt_cmd_vel): 
    clipped = clip_velocities(gt_cmd_vel)
    scaled_cmd_vel = scale_min_max(clipped)    
    return scaled_cmd_vel

def transform_to_gt_scale(cmd_vel_tensor, device):
    return reverse_transform(cmd_vel_tensor).to(device) 