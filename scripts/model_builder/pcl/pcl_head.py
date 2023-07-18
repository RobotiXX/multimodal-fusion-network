# from typing import List

import torch
import torch.nn as nn
from ..image.backbone import make_mlp
from .pointnet_backbone import PclBackbone



class PclMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        self.backbone_pcl = PclBackbone()

        self.common = nn.Sequential(
            nn.Linear(42592,128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU()            
        )

        self.goal_encoder = make_mlp( [4, 64, 16], 'relu', False, False, 0.0)

        self.shared_feat_encod_goal = nn.Sequential(
           nn.Linear(64+16,16),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        
        self.linear_vel = nn.Linear(16,1)

        self.angular_vel =  nn.Sequential(
         nn.Linear(16,8),
         nn.LeakyReLU(),
         nn.Linear(8,1))

    def forward(self, input, goal):

        point_cloud_feat = self.backbone_pcl(input.float())
        feat_shared = self.common(point_cloud_feat)

        goal = self.goal_encoder(goal)

        input = torch.cat([feat_shared, goal],dim=-1)

        goal_encoded_shared_input = self.shared_feat_encod_goal(input)

        x = self.linear_vel(goal_encoded_shared_input)
        y = self.angular_vel(goal_encoded_shared_input)
          
        return x, y




