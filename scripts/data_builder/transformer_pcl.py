import numpy as np
import torch
import torch.nn as nn


def scale_min_max(points,min_val=0, max_val=6):    

    x = np.expand_dims(np.array(points[:,0]), axis=1)    
    y = np.expand_dims(np.array(points[:,1]), axis=1)
    z = np.expand_dims(np.array(points[:,2]), axis=1)

    x = np.interp(x,(0, 8), (min_val, 8))
    y = np.interp(y,(-3, 3), (min_val, max_val))

    return np.concatenate([x,y,z], axis=1)



def get_voxelized_points(points_array):
    # Define the grid dimensions
    grid_size = 122  # Adjust based on your requirements

    # Create an empty voxel grid
    voxel_grid = np.zeros((162, grid_size, 42))    

    # Calculate the voxel size based on the grid dimensions
    voxel_size = 0.05

    if points_array.shape[0] == 0:
        input_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0)
        # print(f'returned_zeroed_array {input_tensor.shape}')
        return input_tensor

    # scale coordinate value
    points_array = scale_min_max(points_array)
    

    # Map points to the voxel grid
    grid_indices = np.floor(points_array / voxel_size).astype(int)    
    grid_indices = np.clip(grid_indices, np.array([0,0,0]), np.array([161,121,41]))

    # unique_indices, counts = np.unique(grid_indices, return_counts=True, axis=0)

    voxel_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]] = 1

    # Convert the voxel grid to a PyTorch tensor
    input_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch and channel dimensions

    return input_tensor 