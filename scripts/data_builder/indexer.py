import pickle
import coloredlogs, logging
import os
from random import shuffle
from torch.utils.data import Dataset
coloredlogs.install()

class IndexDataset(Dataset):

    def __init__(self, dir_path):        
        self.root_path = dir_path
        self.pickle_path = os.path.join(dir_path , 'snapshot.pickle')        
        
        logging.info(f'Parsing pickle file: {self.pickle_path}')
    
        with open(self.pickle_path, 'rb') as data:
            self.content = pickle.load(data)

        logging.info('Picklefile loaded')

        # Exclude keys that does not have a local goal [as the robot did not travel 10 meters]
        keys = list(self.content.keys())
        for key in keys:
            if 'local_goal' not in self.content[key].keys():
                self.content.pop(key)

    def __len__(self):
        # As images and point clouds will be in sets of 4
        return int(len(self.content.keys()) - 4)
    
    def __getitem__(self, offset_index) :
        # We are taking 4 sequential images, point clouds each time to account for temporal variation
        start_index = offset_index
        end_index = start_index + 3

        # Get data from respective index
        prev_cmd_vel = self.content[end_index]['prev_cmd_vel']
        gt_cmd_vel = self.content[end_index]['gt_cmd_vel']
        local_goal = self.content[end_index]['local_goal']
        robot_position = self.content[end_index]['robot_position']
        
        # Image paths
        image_paths = [ os.path.join(self.root_path, str(i)+'.jpg') for i in range(start_index, end_index+1) ]
        
        # only keep points that are under 5 + 1 (delta) meters from the robot
        filtered_points = []
        grnd_pts = []
        # print(list(self.content.keys()), start_index, end_index)
        for point_snapshot in range(start_index, end_index+1):            
            for point in self.content[point_snapshot]['point_cloud']:
                if point[2] >= 0.01 and (point[0]**2 + point[1]**2 + point[2]**2) <= 150:                    
                    filtered_points.append(point)   
                if point[2] < 0.01 and (point[0]**2 + point[1]**2 + point[2]**2) <= 49:                    
                    grnd_pts.append(point)         

        filtered_points = filtered_points[-6000:]

        if len(filtered_points) < 6000:
            shortage = 6000 - len(filtered_points)
            print(f'shortage points: {shortage}')
            filtered_points.extend(grnd_pts[-shortage:])

        return (image_paths, filtered_points, local_goal, prev_cmd_vel, robot_position, gt_cmd_vel)