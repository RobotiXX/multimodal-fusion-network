import pickle
import coloredlogs, logging
import os

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
            if len(self.content[key]['local_goal'])!=12:                
                # print(key)
                self.content.pop(key)

    def __len__(self):
        # As images and point clouds will be in sets of 4
        return int(len(self.content.keys()) - 4)
    
    def __getitem__(self, offset_index) :
        # We are taking 4 sequential images, point clouds each time to account for temporal variation
        start_index = offset_index

        # Get data from respective index       
        gt_cmd_vel = self.content[start_index]['gt_cmd_vel']
        local_goal = self.content[start_index]['local_goal']
        robot_position = self.content[start_index]['robot_position']
        
        # Image paths
        image_paths = [ os.path.join(self.root_path, str(i)+'.jpg') for i in range(start_index, start_index+1) ]
        
        # only keep points that are under 5 + 1 (delta) meters from the robot
        point_clouds = []
        # print(list(self.content.keys()), start_index, end_index)
        for point_snapshot in range(start_index, start_index+1):
            filtered_points = []            
            for point in self.content[point_snapshot]['point_cloud']:
                if point[0] >= 0 and point[0] <= 8.009 and point[1]>=-3 and point[1]<=3 and point[2] >= 0.0299 and point[2] <= 2.0299:
                    filtered_points.append(point)                           

            point_clouds.append(filtered_points)                

        return (image_paths, point_clouds, local_goal, robot_position, gt_cmd_vel)