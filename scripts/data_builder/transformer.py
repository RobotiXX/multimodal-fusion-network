import torch
from torchvision import transforms
import pickle
import numpy as np
import coloredlogs, logging
import os
import cv2
import tf

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

coloredlogs.install()




def read_images(path):
    # print(f"{path = }")
    image = cv2.imread(path)
    # Will have to do some re-sizing
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_transformation_matrix(position, quaternion):
    
    q = tf.transformations.quaternion_from_euler(quaternion[0], quaternion[1], quaternion[2])
    rotation_matrix = tf.transformations.quaternion_matrix(q)
    translation = -np.matmul(rotation_matrix, np.array([position[0],position[1],position[2],1]).reshape(4,1))
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
        self.local_goal = data[2]
        self.prev_cmd_vel = data[3]        
        self.robot_position  = data[4]
        self.gt_cmd_vel = data[5]
        

        images = [ self.image_transforms(read_images(path)) for path in self.image_paths]
        stacked_images = torch.cat(images, dim=0)
        
        # Transform local goal into robot frame
        tf_matrix = get_transformation_matrix(self.robot_position[0],self.robot_position[1])
        local_goal_rbt_frame = np.matmul(tf_matrix, np.array([self.local_goal[0], self.local_goal[1],0,1]).reshape(4,1))
        
        local_goal = (local_goal_rbt_frame[0,0], local_goal_rbt_frame[1,0], 0, 0)
        # robot_curr_pos = (0, robot_curr_pos[1,0])
        # Transform point-clouds to 3D-Cylider co-ordinate system
        # x= np.matmul(tf_matrix, np.array([self.robot_position[0][0], self.robot_position[0][1],self.robot_position[0][2],1]).reshape(4,1))
        # print(f'matrix: {tf_matrix}')
        # print(f'robot pos: {self.robot_position}') 
        # print(f'local_goal: {self.local_goal}')
        # print(f'matmul: {x}')
        # print(f'tf_local_goal: {local_goal}')
        point_clouds = np.array(self.point_clouds)   
        # if point_clouds.shape[0] != 6000:
        #     print(f'pint_cloud {point_clouds.shape}')

        
        # print(f'gt_velocity: {self.gt_cmd_vel}')

        prev_cmd_vel = np.array(self.prev_cmd_vel, dtype=np.float64)
        perv_linear = np.expand_dims(prev_cmd_vel[:,0], axis=1)
        prev_anglular = np.expand_dims(prev_cmd_vel[:,2], axis=1)
        # print(prev_cmd_vel)
        lin_and_angular = np.concatenate([perv_linear, prev_anglular], axis=1)
        # print(lin_and_angular.shape)
        # gt_cmd_vel = (100 * self.gt_cmd_vel[0], 5000 * np.around(self.gt_cmd_vel[2], 3))
        gt_cmd_vel = (self.gt_cmd_vel[0], np.around(self.gt_cmd_vel[2], 2))
        local_goal = torch.tensor(local_goal, dtype=torch.float32).ravel()

        prev_cmd_vel = torch.tensor(lin_and_angular, dtype=torch.float32).ravel()
        gt_cmd_vel = torch.tensor(gt_cmd_vel, dtype=torch.float32).ravel()

        point_clouds =  torch.tensor(point_clouds)
        point_clouds = torch.t(point_clouds)

        return (stacked_images, point_clouds, local_goal, prev_cmd_vel, gt_cmd_vel)



