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

        self.backbone_pcl = PclBackbone().float()

        self.common = nn.Sequential(
            nn.Linear(63888+128, 512),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU()            
        )

        self.goal_encoder = make_mlp( [4, 64, 128], 'relu', False, False, 0.0)
        
        self.linear_vel = nn.Linear(64,1)

        self.angular_vel =  nn.Sequential(
         nn.Linear(64,32),
         nn.LeakyReLU(),
         nn.Linear(32,1))

    def forward(self, input, goal):
        
        
        point_cloud_feat = self.backbone_pcl(input.float())
        goal = self.goal_encoder(goal)

        point_cloud_feat = torch.cat([point_cloud_feat, goal],dim=-1)

        feat_shared = self.common(point_cloud_feat)
        

        x = self.linear_vel(feat_shared)
        y = self.angular_vel(feat_shared)
          
        return x, y




