# from typing import List

import torch
import torch.nn as nn
from ..image.backbone import make_mlp
from ..image.backbone_fusion import ImageFusionModel
from ..pcl.pointnet_fusion import PointNetDenseFusionModel
from ..pcl.pointnet_backbone import PclBackbone
from .fustion_mlp import FusionMLP
from .pcl_mlp import PclMLP
from .image_mlp import ImageMLP


class BcFusionModel(nn.Module):
    def __init__(
        self,        
        backbone_img: str = "resnet18",
        sparse_shape = [480, 360, 32],
        controller_encoder: list = [(198256+512), 128, 64, 32, 2],
        goal_encoder: list = [4, 256, 128],
        prev_cmd_encoder: list = [2 * 20, 64, 128],
        n_frames: int = 4,
        n_channels: int = 3,
        act: str = 'relu',
        l_act: bool = False, bn: bool = True, dropout: float = 0.0
    ):

        super().__init__()
        
        self.backbone_pcl = PclBackbone()
        self.backbone_img = ImageFusionModel(backbone=backbone_img, n_frames= n_frames, n_channels=n_channels)
        self.goal_encoder = make_mlp(goal_encoder, act, l_act, False, dropout)
        # self.prev_cmd_encoder = make_mlp(prev_cmd_encoder, act, l_act, False, dropout)

        self.mx_pool_lyr2_img = nn.MaxPool2d(kernel_size= 4, stride=3)
        self.mx_pool_lyr3_img = nn.MaxPool2d(kernel_size= 2, stride=3)
        self.mx_pool_lyr4_img = nn.MaxPool2d(kernel_size= 2, stride=2)

        self.mx_pool_lyr1_pcl = nn.MaxPool1d(kernel_size= 12, stride=  8)
        self.mx_pool_lyr2_pcl = nn.MaxPool1d(kernel_size= 8, stride= 6)
        self.mx_pool_lyr3_pcl = nn.MaxPool1d(kernel_size= 2, stride= 2)

        self.controller_img = ImageMLP()
        self.controller_pcl = PclMLP()
        self.fusion_mlp = FusionMLP()
        

    def forward(self, stacked_images, pcl, local_goal, prev_cmd_vel):


        imgs_feat, intr_img_rep = self.backbone_img(stacked_images)
        point_cloud_feat, intr_pts_rep = self.backbone_pcl(pcl.float())
        
        goal = self.goal_encoder(local_goal)
        # prev_cmd = self.prev_cmd_encoder(prev_cmd_vel)

        pooled_lyr1_img = self.mx_pool_lyr2_img( intr_img_rep['layer2'] )
        pooled_lyr2_img = self.mx_pool_lyr3_img( intr_img_rep['layer3'] )
        pooled_lyr3_img = self.mx_pool_lyr4_img( intr_img_rep['layer4'] )

        pooled_lyr1_img = pooled_lyr1_img.contiguous().view(pooled_lyr1_img.shape[0], -1)
        pooled_lyr2_img = pooled_lyr2_img.contiguous().view(pooled_lyr2_img.shape[0], -1)
        pooled_lyr3_img = pooled_lyr3_img.contiguous().view(pooled_lyr3_img.shape[0], -1)
        

        pooled_lyr1_pcl = intr_pts_rep[0]
        pooled_lyr2_pcl = intr_pts_rep[1]
        pooled_lyr3_pcl = point_cloud_feat

        # print(f'pcl:{pooled_lyr1_pcl.shape}, image:{pooled_lyr1_img.shape}')
        # print(f'pcl:{pooled_lyr2_pcl.shape}, image:{pooled_lyr2_img.shape}')
        # print(f'pcl:{pooled_lyr3_pcl.shape}, image:{pooled_lyr3_img.shape}')

        fusion_input1 = torch.cat([pooled_lyr1_img, pooled_lyr1_pcl], dim=1)
        fusion_input2 = torch.cat([pooled_lyr2_img, pooled_lyr2_pcl], dim=1)
        fusion_input3 = torch.cat([pooled_lyr3_img, pooled_lyr3_pcl], dim=1)



        predict_fusion_model_lin, predict_fusion_model_angular = self.fusion_mlp(fusion_input1, fusion_input2, fusion_input3, goal)

        input_image_features = torch.cat([imgs_feat, goal],dim=-1)
        img_lin, img_angular = self.controller_img(input_image_features)

        input_pcl_features = torch.cat([pooled_lyr1_pcl, goal],dim=-1)
        pcl_lin, pcl_angular = self.controller_pcl(input_pcl_features)


        return predict_fusion_model_lin, predict_fusion_model_angular, img_lin, img_angular, pcl_lin, pcl_angular
