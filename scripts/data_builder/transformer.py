import torch
from torchvision import transforms
import pickle
import numpy as np
import coloredlogs, logging
import os
import cv2
# import tf
import pyquaternion as pq

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from .transformer_pcl import get_voxelized_points
from .gaussian_weights import get_gaussian_weights
from .cmd_scaler import transform_cmd_vel


coloredlogs.install()




def read_images(path):
    # print(f"{path = }")
    image = cv2.imread(path)
    # Will have to do some re-sizing
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_transformation_matrix(position, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    robo_coordinate_in_glob_frame  = np.array([[np.cos(theta), -np.sin(theta), position[0]],
                    [np.sin(theta), np.cos(theta), position[1]],
                    [0, 0, 1]])
    return robo_coordinate_in_glob_frame


class ApplyTransformation(Dataset):
    def __init__(self, input_data, grid_size = [72, 30, 30]):
        self.grid_size = np.asarray(grid_size)  
        self.input_data = input_data    
        self.image_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224,224),antialias=True)
            ])
    
    def __len__(self):
         # TODO: this will return 1 example set with the following details
        return len(self.input_data)

    def __getitem__(self, index):
        # Transform images
        data = self.input_data[index]
        self.image_paths = data[0]
        self.point_clouds = data[1]
        self.way_pts = data[2]        
        self.robot_position  = data[3]
        self.gt_cmd_vel = data[4]

        images = [ self.image_transforms(read_images(path)) for path in self.image_paths]
        stacked_images = torch.cat(images, dim=0)
        
        # Transform local goal into robot frame
        tf_matrix = get_transformation_matrix(self.robot_position[0],self.robot_position[1])     
        tf_inverse = np.linalg.pinv(tf_matrix)

        goals = np.concatenate([ np.array(self.way_pts), np.ones((12,1))], axis=1).transpose()
        

        all_pts = np.matmul(tf_inverse, goals) * get_gaussian_weights(2,1.3)
        all_pts = all_pts[:2, :]

        way_pts = all_pts[:, :4]
        local_goal = all_pts[:, 4]

        # print(f'{all_pts/150}')
        # print(f'{local_goal}')

        point_clouds = np.array(self.point_clouds[0])   
        point_clouds = get_voxelized_points(point_clouds)

        gt_cmd_vel = np.array([self.gt_cmd_vel[0], self.gt_cmd_vel[2]])
        gt_cmd_vel = np.around(gt_cmd_vel,2)
        gt_cmd_vel = np.expand_dims(gt_cmd_vel, axis=0)
        gt_cmd_vel = transform_cmd_vel(gt_cmd_vel)

        gt_pts = torch.tensor(way_pts, dtype=torch.float32).ravel()

        local_goal = torch.tensor(local_goal, dtype=torch.float32).ravel()        

        gt_cmd_vel = torch.tensor(gt_cmd_vel, dtype=torch.float32).ravel()        

        return (stacked_images, point_clouds, local_goal, gt_pts, gt_cmd_vel)



