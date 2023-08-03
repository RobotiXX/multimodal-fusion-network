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

        self.pcl_weights = torch_load_weights('/scratch/bpanigr/fusion-network/pcl_backbone_changed_model_at_100_0.08454692389459491.pth')
        self.image_weights = torch_load_weights('/home/bpanigr/Workspace/rnn_gw_img_way_pts_model_at_140.pth')
        
        self.image.load_state_dict(self.image_weights, strict=False)
        self.pcl.load_state_dict(self.pcl_weights, strict=False)

        set_trainable_false(self.image)
        set_trainable_false(self.pcl)

        self.modality_fusion_layer = nn.Sequential(
            nn.Linear(1024+512,2048),
            nn.ELU(),
            nn.Linear(2048,2304),
            nn.ELU(),
            nn.Linear(2304,1024),
            nn.ELU()
        )

        self.global_path_fusion = nn.Sequential(
            nn.Linear(44,64),
            nn.ELU(),
            nn.Linear(64,128),
            nn.ELU()
        )

        self.joint_perception_path_feautres = nn.Sequential(
            nn.Linear(128+1024,1024),
            nn.ELU()
        )

        self.predict = nn.Linear(1024,2)

    def forward(self, stacked_images, pcl, local_goal):
        

        rnn_img_out, final_img_feat = self.image(stacked_images, local_goal)
        rnn_pcl_out, final_pcl_feat = self.pcl(pcl, local_goal)

        backbone_feats = torch.cat([rnn_pcl_out, rnn_img_out], dim=-1)
        fustion_features = self.modality_fusion_layer(backbone_feats)        
        

        second_layer_features = torch.cat([final_pcl_feat,final_img_feat], dim=-1)
        global_path_encoding = self.global_path_fusion(second_layer_features)
        

        final_features_concat = torch.cat([global_path_encoding,fustion_features], dim=-1)
        final_feat = self.joint_perception_path_feautres(final_features_concat)

        prediction = self.predict(final_feat)

        return prediction
