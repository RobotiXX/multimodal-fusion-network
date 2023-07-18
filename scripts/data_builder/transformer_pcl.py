import numpy as np
import torch
import torch.nn as nn


def scale_min_max(points,min_val=0, max_val=6):    

    x = np.expand_dims(np.array(points[:,0]), axis=1)    
    y = np.expand_dims(np.array(points[:,1]), axis=1)
    z = np.expand_dims(np.array(points[:,2]), axis=1)

    x = np.interp(x,(-1, 5), (min_val, max_val))
    y = np.interp(y,(-3, 3), (min_val, max_val))

    return np.concatenate([x,y,z], axis=1)



def get_voxelized_points(points_array):
    # Define the grid dimensions
    grid_size = 201  # Adjust based on your requirements

    # Calculate the voxel size based on the grid dimensions
    voxel_size = 0.03

    # scale coordinate value
    points_array = scale_min_max(points_array)

    # Create an empty voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size))

    # Map points to the voxel grid
    grid_indices = np.floor(points_array / voxel_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, grid_size - 1)
    voxel_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]] = 1

    # Convert the voxel grid to a PyTorch tensor
    input_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch and channel dimensions

    return input_tensor 