import numpy as np
import torch
import torch.nn as nn


def get_voxelized_points(points_array):
    # Define the grid dimensions
    grid_size = 125  # Adjust based on your requirements

    # Calculate the voxel size based on the grid dimensions
    voxel_size = 0.08

    # Calculate the minimum coordinate value
    min_coord = np.min(points_array)

    # Create an empty voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size))

    # Map points to the voxel grid
    grid_indices = np.floor((points_array - min_coord) / voxel_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, grid_size - 1)
    voxel_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]] = 1

    # Convert the voxel grid to a PyTorch tensor
    input_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch and channel dimensions

    return input_tensor 