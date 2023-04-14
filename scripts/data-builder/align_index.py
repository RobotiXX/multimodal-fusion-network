import torch
import pickle
import numpy as np
import coloredlogs, logging
import os

from torch.utils.data import Dataset
coloredlogs.install()

class data_preprocessing(Dataset):

    def __init__(self, dir_path):        
        self.path = os.path.join(dir_path , 'snapshot.pickle')        
        
        logging.info('Parsing pickle file...')

        with open(self.path, 'rb') as data:
            self.content = pickle.load(data)

        logging.info('Picklefile loaded')

    def __len__(self):
        # Excluding last 2 minutes of recording
        # Snapshot is taken at 2 frames per second
        return (len(self.keys()) - 244) / 4
    
    def __getitem__(self, offset_index) :
        # We are taking 4 sequential images, point clouds each time to account for temporal variation
        start_index = offset_index * 4
        end_index = start_index + 3

        # Get data from respective index
        prev_cmd_vel = self.content[end_index]
        gt_cmd_vel = self.content[end_index]
        local_goal = self.content[end_index]
        
        # Image paths
        image_paths = [ os.path.join(self.dir_path, str(i), '.jpg') for i in range(start_index, end_index+1) ]
        
        # only keep points that are under 5 meters from the robot
        point_clouds = []
        for point_snapshot in range(start_index, end_index+1):
            filtered_points = []
            for point in self.content[point_snapshot]['point_cloud']:
                if (point[0]**2 + point[1]**2 + point[2]**2 ) <= 26:
                    filtered_points.append(point)
            point_clouds.append(filtered_points)
        
        return image_paths, point_clouds, local_goal, prev_cmd_vel, gt_cmd_vel

        



