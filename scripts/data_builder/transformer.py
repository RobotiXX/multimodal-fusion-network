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
    def __init__(self, input_data, grid_size = [72, 30, 30]):
        self.grid_size = np.asarray(grid_size)  
        self.input_data = input_data    
        self.image_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224,224),antialias=True),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean hardcoded
                    #                       std=[0.229, 0.224, 0.225])  # ImageNet std hardcoded
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
        robot_coordinate_in_glob_frame = get_transformation_matrix(self.robot_position[0],self.robot_position[1])
        transform_to_robot_coordinate =   np.linalg.pinv(robot_coordinate_in_glob_frame)

        local_goal_in_robot_frame = transform_to_robot_coordinate @ np.asanyarray([self.local_goal[0], self.local_goal[1],1]).reshape((3,1))	
        local_goal = (local_goal_in_robot_frame[0,0], local_goal_in_robot_frame[1,0])

        # Transform point-clouds to 3D-Cylider co-ordinate system
        point_clouds = np.concatenate(self.point_clouds, axis=0)   

        # point_clouds = point_clouds[-25000, :]
        # TODO: subsample the point clouds to keep a fixed number of points across frames
        # xyz_polar = cart2polar(point_clouds)

        # max_bound_r = np.percentile(xyz_polar[:, 0], 100, axis=0)
        # min_bound_r = np.percentile(xyz_polar[:, 0], 0, axis=0)

        # max_bound = np.max(xyz_polar[:, 1:], axis=0)
        # min_bound = np.min(xyz_polar[:, 1:], axis=0)

        # max_bound = np.concatenate(([max_bound_r],max_bound))
        # min_bound = np.concatenate(([min_bound_r], min_bound))

        # range_to_crop = max_bound - min_bound
        # cur_grid_size = (self.grid_size - 1)
        # intervals = range_to_crop / cur_grid_size

        # if (intervals == 0).any(): print("Zero interval!")
        # grid_index = (np.floor(( np.clip(xyz_polar, min_bound, max_bound) - min_bound) / intervals)).astype(int)
        
        # # Center data around each voxel for PTnet
        # voxel_centers = (grid_index.astype(np.float32) + 0.5) * intervals + min_bound
        # return_xyz = xyz_polar - voxel_centers
        # transformed_pcl = np.concatenate((return_xyz, xyz_polar, point_clouds[:, :2]), axis=1)
        
        gt_cmd_vel = (self.gt_cmd_vel[1]*1000)
        local_goal = torch.tensor(local_goal, dtype=torch.float32).ravel()

        prev_cmd_vel = torch.tensor(self.prev_cmd_vel, dtype=torch.float32).ravel()
        gt_cmd_vel = torch.tensor(gt_cmd_vel, dtype=torch.float32).ravel()

        point_clouds =  torch.tensor(point_clouds)
        point_clouds = torch.t(point_clouds)

        return (stacked_images, point_clouds, local_goal, prev_cmd_vel, gt_cmd_vel)




