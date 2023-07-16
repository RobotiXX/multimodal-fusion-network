import torch
from torchvision import transforms
import pickle
import numpy as np
import coloredlogs, logging
import os
import cv2

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

coloredlogs.install()

def read_images(path):
    # print(f"{path = }")
    image = cv2.imread(path)
    # Will have to do some re-sizing
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_transformation_matrix(position, quaternion):
    theta = R.from_quat([quaternion.x, quaternion.y, quaternion.z, quaternion.w]).as_euler('XYZ')[2]
    robo_coordinate_in_glob_frame  = np.array([[np.cos(theta), -np.sin(theta), position.x],
                    [np.sin(theta), np.cos(theta), position.y],
                    [0, 0, 1]])
    return robo_coordinate_in_glob_frame

def cart2polar(xyz):
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    theta =  np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack((r,theta, xyz[:,2]), axis=1)


class ApplyTransformation(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data    
        self.image_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224,224),antialias=True),
            ])
    
    def __len__(self):
         # TODO: this will return 1 example set with the following details
        return 1

    def __getitem__(self, index):
        # Transform images
        data = self.input_data
        self.images = data['images']
        self.point_clouds = data['pcl']
        self.local_goal = data['local_goal']
        self.prev_cmd_vel = data['prev_cmd_vel']        
        self.robot_position  = data['robot_pos']
        # self.gt_cmd_vel = data[5]
        

        images = [ self.image_transforms(img) for img in self.images]
        stacked_images = torch.cat(images, dim=0)
        
        # Transform local goal into robot frame
    

        # Transform point-clouds to 3D-Cylider co-ordinate system
        point_clouds = np.concatenate(self.point_clouds, axis=0)   

        robot_coordinate_in_glob_frame = get_transformation_matrix(self.robot_position[0],self.robot_position[1])
        transform_to_robot_coordinate =   np.linalg.pinv(robot_coordinate_in_glob_frame)

        local_goal_in_robot_frame = transform_to_robot_coordinate @ np.asanyarray([self.local_goal[0], self.local_goal[1],1]).reshape((3,1))
        robot_curr_pos = transform_to_robot_coordinate @ np.asanyarray([self.robot_position[0].x, self.robot_position[0].y,1]).reshape((3,1))		
        
        robot_curr_pos = (robot_curr_pos[0,0], robot_curr_pos[1,0])
        
        local_goal = (local_goal_in_robot_frame[0,0], local_goal_in_robot_frame[1,0])            
        
        print(f'transformed: {local_goal}')
        print(f'robotpos transformed: {robot_curr_pos}')

        local_goal = torch.tensor(local_goal, dtype=torch.float32).ravel()

        prev_cmd_vel = torch.tensor(self.prev_cmd_vel, dtype=torch.float32).ravel()

        point_clouds =  torch.tensor(point_clouds)
        point_clouds = torch.t(point_clouds)

        return (stacked_images, point_clouds, local_goal, prev_cmd_vel)




