# from typing import List

import torch
import torch.nn as nn
from ..image.backbone import make_mlp, get_backbone
from ..pcl.pointnet import PointNetDenseCls



class BCModelPcl(nn.Module):
    def __init__(
        self,        
        backbone_img: str = "resnet18",
        sparse_shape = [480, 360, 32],
        controller_encoder: list = [(198256+512), 256, 128, 64, 2],
        goal_encoder: list = [2, 256, 128],
        prev_cmd_encoder: list = [2 * 20, 64, 128],
        n_frames: int = 4,
        n_channels: int = 3,
        act: str = 'relu',
        l_act: bool = False, bn: bool = False, dropout: float = 0.0
    ):

        super().__init__()
        self.backbone_pcl = PointNetDenseCls().float()
        self.backbone_img = get_backbone(backbone_img, n_frames, n_channels)
        self.controller = make_mlp(controller_encoder, act, l_act, bn, dropout)
        self.goal_encoder = make_mlp(goal_encoder, act, l_act, bn, dropout)
        self.prev_cmd_encoder = make_mlp(prev_cmd_encoder, act, l_act, bn, dropout)

    def forward(self, pcl, stacked_images,local_goal, prev_cmd_vel):
        # print(f"{img.stacked_images = }")
        # image features in shape (B, 512)
        imgs_feat = self.backbone_img(stacked_images)
        point_cloud_feat = self.backbone_pcl(pcl.float())
        # goal features in shape (B, 128)
        goal = self.goal_encoder(local_goal)
        # prec_cmd_vel features in shape (B, 128)
        # print(prev_cmd_vel.shape)
        prev_cmd = self.prev_cmd_encoder(prev_cmd_vel)
        # concat all encoded features together along the last dim,
        # making a tensor of shape (B, 512 + 128 + 128) = (B, 768)
        # print(point_cloud.shape)
        # print(goal.shape)
        # print(prev_cmd.shape)
        multimodal_feat = torch.cat([point_cloud_feat,imgs_feat], dim=1)
        features = torch.cat([multimodal_feat, goal, prev_cmd], dim=-1)
        # return the action in shape (B, 2)
        return self.controller(features)
