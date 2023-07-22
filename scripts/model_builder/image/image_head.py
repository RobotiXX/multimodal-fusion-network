# from typing import List

import torch
import torch.nn as nn
from .backbone_fusion import ImageFusionModel
from .backbone import make_mlp



class ImageHeadMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()
        self.backbone = ImageFusionModel()
        self.goal_encoder = make_mlp( [4, 64, 128], 'relu', False, False, 0.0)


        self.common = nn.Sequential(
            nn.Linear(512+128,128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.linear_vel = nn.Linear(64,1)
        
        self.angular_vel = nn.Sequential(            
            nn.Linear(64,32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,1)            
        )

    def forward(self, input, goal):
        
        image_features, fusion_feat = self.backbone(input)                
        goal = self.goal_encoder(goal)

        img_cloud_feat = torch.cat([image_features, goal],dim=-1)

        feat_shared_common = self.common(img_cloud_feat)

        x = self.linear_vel(feat_shared_common)
        y = self.angular_vel(feat_shared_common)
          
        return x, y




