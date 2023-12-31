# from typing import List

import torch
import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights
from .backbone import Backbone
from ..image.backbone import make_mlp
# from .feature_encoder import FeatureEncoder
# from .spconv import Asymm_3d_spconv
from .pointnet import PointNetDenseCls



class BCModelPcl(nn.Module):
    def __init__(
        self,        
        sparse_shape = [480, 360, 32],
        controller_encoder: list = [216256, 256, 128, 64, 2],
        goal_encoder: list = [2, 256, 128],
        prev_cmd_encoder: list = [2 * 18, 64, 128],
        n_frames: int = 4,
        n_channels: int = 3,
        act: str = 'relu',
        l_act: bool = False, bn: bool = False, dropout: float = 0.0
    ):

        super().__init__()
        self.backbone = PointNetDenseCls().float()
        self.controller = make_mlp(controller_encoder, act, l_act, bn, dropout)
        self.goal_encoder = make_mlp(goal_encoder, act, l_act, bn, dropout)
        self.prev_cmd_encoder = make_mlp(prev_cmd_encoder, act, l_act, bn, dropout)

    def forward(self, pcl, local_goal, prev_cmd_vel):
        # print(f"{img.stacked_images = }")
        # image features in shape (B, 512)
        point_cloud = self.backbone(pcl.float())
        # goal features in shape (B, 128)
        goal = self.goal_encoder(local_goal)
        # prec_cmd_vel features in shape (B, 128)
        prev_cmd = self.prev_cmd_encoder(prev_cmd_vel)
        # concat all encoded features together along the last dim,
        # making a tensor of shape (B, 512 + 128 + 128) = (B, 768)
        print(point_cloud.shape)
        print(goal.shape)
        print(prev_cmd.shape)
        features = torch.cat([point_cloud, goal, prev_cmd], dim=-1)
        # return the action in shape (B, 2)
        return self.controller(features)
