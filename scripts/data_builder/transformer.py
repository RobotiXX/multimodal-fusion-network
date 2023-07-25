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

coloredlogs.install()




def read_images(path):
    # print(f"{path = }")
    image = cv2.imread(path)
    # Will have to do some re-sizing
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_transformation_matrix(position, quaternion):
    q_normalized = quaternion / np.linalg.norm(quaternion)
    rotation_matrix = R.from_quat(q_normalized).as_matrix()    
    translation = -np.matmul(rotation_matrix, np.array([position[0],position[1],0]).reshape(3,1))
    transformation_matrix = np.concatenate([rotation_matrix[:,:3], translation], axis=1)    
    return transformation_matrix

def cart2polar(xyz):
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    theta =  np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack((r,theta, xyz[:,2]), axis=1)


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
        goals = np.concatenate([ np.array(self.way_pts), np.zeros((6,1)), np.ones((6,1))], axis=1).transpose()

        all_pts = np.matmul(tf_matrix, goals) * 100
        all_pts = all_pts[:2, :]

        way_pts = all_pts[:, :-1]
        local_goal = all_pts[:, -1]

        # print(f'{way_pts.shape}')
        # print(f'{local_goal.shape}')

        point_clouds = np.array(self.point_clouds[0])   
        point_clouds = get_voxelized_points(point_clouds)

        gt_cmd_vel = (100 * self.gt_cmd_vel[0], 1050 * np.around(self.gt_cmd_vel[2], 3))

        
        gt_pts = torch.tensor(way_pts, dtype=torch.float32).ravel()

        local_goal = torch.tensor(local_goal, dtype=torch.float32).ravel()        

        gt_cmd_vel = torch.tensor(gt_cmd_vel, dtype=torch.float32).ravel()        

        return (stacked_images, point_clouds, local_goal, gt_pts, gt_cmd_vel)



