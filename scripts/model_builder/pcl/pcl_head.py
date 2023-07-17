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
            nn.Linear(55296+128,1024),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(512,128),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(128,32),
            # nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.goal_encoder = make_mlp( [4, 256, 128], 'relu', False, False, 0.0)

        self.linear_vel = nn.Linear(32,1)
        self.angular_vel = nn.Linear(32,1)

    def forward(self, input, goal):

        point_cloud_feat = self.backbone_pcl(input.float())
        

        goal = self.goal_encoder(goal)
        input = torch.cat([point_cloud_feat, goal],dim=-1)

        feat_shared = self.common(input)

        x = self.linear_vel(feat_shared)
        y = self.angular_vel(feat_shared)
          
        return x, y




