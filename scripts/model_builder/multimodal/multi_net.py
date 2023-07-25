# from typing import List

import torch
import torch.nn as nn
from ..pcl.pcl_head import PclMLP
from ..image.image_head import ImageHeadMLP

def set_trainable_false(model):
    for param in model.parameters():
        param.requires_grad = False    

def torch_load_weights(path):
    check_point = torch.load(path)
    model_weights = check_point['model_state_dict']
    return model_weights

class MultiModalNet(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.image =  ImageHeadMLP()        
        self.pcl =  PclMLP()

        self.image_weights = torch_load_weights('/home/bpanigr/Workspace/pre_img_way_pts_model_at_110.pth')
        self.pcl_weights = torch_load_weights('/scratch/bpanigr/fusion-network/way_pts2_model_at_120_0.013270827467591461.pth')

        del self.pcl_weights['previous.2.weight']
        del self.pcl_weights['previous.2.bias']
        
        self.image.load_state_dict(self.image_weights, strict=False)
        self.pcl.load_state_dict(self.pcl_weights, strict=False)

        set_trainable_false(self.image)
        set_trainable_false(self.pcl)

        self.fusion_featuers = nn.Sequential(
            nn.Linear(63888+512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()

        )

        self.intermediate_features = nn.Sequential(
            nn.Linear(256+128+128,256),
            nn.LeakyReLU()
        )

        self.goal_encoded_features = nn.Sequential(
            nn.Linear(256+256+64+128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )

        self.predict = nn.Linear(256,2)

    def forward(self, stacked_images, pcl, local_goal):

        image_features, image_feat_encoded, image_feat_with_goal = self.image(stacked_images, local_goal)
        pcl_features, pcl_feat_encoded, pcl_feat_with_goal = self.pcl(pcl, local_goal)

        backbone_feats = torch.cat([image_features, pcl_features], dim=-1)
        
        fustion_features = self.fusion_featuers(backbone_feats)

        second_layer_features = torch.cat([fustion_features,image_feat_encoded, pcl_feat_encoded], dim=-1)
        intermediate_features = self.intermediate_features(second_layer_features)

        third_layer_features = torch.cat([fustion_features, intermediate_features, image_feat_with_goal,pcl_feat_with_goal], dim=-1)
        goal_encoded_features = self.goal_encoded_features(third_layer_features)

        prediction = self.predict(goal_encoded_features)

        return prediction
