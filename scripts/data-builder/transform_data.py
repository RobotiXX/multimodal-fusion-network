import torch
import pickle
import numpy as np
import coloredlogs, logging
import os
import cv2

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

coloredlogs.install()

class transform_data(Dataset):
    def __init__(self, image_paths, point_clouds, local_goal, prev_cmd_vel, robot_position, gt_cmd_vel, grid_size):
        self.image_paths = image_paths
        self.point_clouds = point_clouds
        self.local_goal = local_goal
        self.prev_cmd_vel = prev_cmd_vel
        self.robot_position  = robot_position
        self.gt_cmd_vel = gt_cmd_vel
        self.grid_size = grid_size        
    
    def get_transformation_matrix(position, orientation):
        theta = R.from_quat(orientation).as_euler('XYZ')[2]
        robo_coordinate_in_glob_frame  = np.array([[np.cos(theta), -np.sin(theta), position.x],
					 [np.sin(theta), np.cos(theta), position.y],
					 [0, 0, 1]])
        return robo_coordinate_in_glob_frame

    def read_images(path):
        image = cv2.imread(path)
        # Will have to do some re-sizing
        return cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
    
    def __len__(self):
         # TODO: this will return 1 example set with the following details
        return 1

    def __getitem__(self, index):
        # Transform images
        images = [self.read_images(path) for path in self.image_paths]

        # Transform local goal into robot frame
        robot_coordinate_in_glob_frame = self.get_transformation_matrix(self.robot_position[0],self.robot_position[1])
        transform_to_robot_coordinate =   np.linalg.pinv(robot_coordinate_in_glob_frame)

        local_goal_coordinate_in_glob_frame = self.get_transformation_matrix(self.local_goal[0],self.local_goal[1])
        local_goal_in_robot_frame = transform_to_robot_coordinate @ local_goal_coordinate_in_glob_frame
        # grab only two points from the local_goal_in_robot_frame  matrix

        # Transform point clouds to 3D-Cylider co-ordinate system











