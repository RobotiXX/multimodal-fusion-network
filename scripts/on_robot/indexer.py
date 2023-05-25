import pickle
import os

from torch.utils.data import Dataset

class IndexDataset(Dataset):

    def __init__(self, data):        

        self.content = data

    def __len__(self):
        # As images and point clouds will be in sets of 4
        return int( (len(self.content.keys())-1) / 4)
    
    def __getitem__(self, offset_index) :
        # We are taking 4 sequential images, point clouds each time to account for temporal variation
        start_index = offset_index * 4
        end_index = start_index + 3

        # Get data from respective index
        prev_cmd_vel = self.content[end_index]['prev_cmd_vel']
        local_goal = self.content[end_index]['local_goal']
        
        # Image paths
        image_paths = self.content['images']
        
        # only keep points that are under 5 + 1 (delta) meters from the robot
        point_clouds = []
        # print(list(self.content.keys()), start_index, end_index)
        for point_snapshot in range(start_index, end_index+1):
            filtered_points = []
            for point in self.content[point_snapshot]['point_cloud']:
                if (point[0]**2 + point[1]**2 + point[2]**2) <= 49:
                    filtered_points.append(point)
            point_clouds.append(filtered_points[:5500])                


        return (image_paths, point_clouds, local_goal, prev_cmd_vel)