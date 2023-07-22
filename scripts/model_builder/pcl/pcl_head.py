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
            nn.Linear(63888, 512),
            nn.LeakyReLU(),            
            nn.Linear(512,256),
            nn.LeakyReLU()                   
        )

        self.previous = nn.Sequential(
            nn.Linear(64+256,128),
            nn.LeakyReLU()            
        )

        self.goal_encoder = make_mlp( [2, 64, 128, 64], 'relu', False, False, 0.0)
        
        # self.linear_vel = nn.Linear(64,1)

        self.angular_vel =  nn.Sequential(
         nn.Linear(128,64),
         nn.LeakyReLU(),
         nn.Linear(64,10))

    def forward(self, input, goal):
        
        
        point_cloud_feat = self.backbone_pcl(input.float())
        # print(f'point cloud: {point_cloud_feat.shape}')
        goal = self.goal_encoder(goal)        

        feat_shared = self.common(point_cloud_feat)
        feat_shared = torch.cat([feat_shared, goal],dim=-1)

        final_feat = self.previous(feat_shared)
        
        y = self.angular_vel(final_feat)
          
        return y



