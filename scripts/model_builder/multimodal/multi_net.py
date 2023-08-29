# from typing import List

import torch
import torch.nn as nn
from ..pcl.pcl_head import PclMLP
from ..image.image_head import ImageHeadMLP
from .tf_model import CustomTransformerModel
from ..image.backbone import make_mlp

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
        self.transformer = CustomTransformerModel()


        self.modality_fusion_layer = nn.Sequential(
            nn.Linear(1024+1024,2304),
            nn.ELU(),
            nn.Linear(2304,1024),
            nn.ELU()
        )

        self.goal_encoder = make_mlp( [2, 128, 64], 'relu', False, False, 0.0)

        self.global_path_predictor = nn.Sequential(
            nn.Linear(1024+64, 512),
            nn.ELU(),
            nn.Linear(512,8),            
        )

        self.global_path_encoder = nn.Sequential(
            nn.Linear(8, 256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU()
        )

        self.joint_perception_path_feautres = nn.Sequential(
            nn.Linear(128+1024,512),
            nn.ELU()
        )

        self.predict_vel = nn.Linear(512,2)        

    def forward(self, stacked_images, pcl, local_goal):
        

        rnn_img_out, img_path, img_vel = self.image(stacked_images, local_goal)
        rnn_pcl_out, pcl_path, pcl_vel = self.pcl(pcl, local_goal)

        backbone_feats = torch.cat([rnn_pcl_out, rnn_img_out], dim=-1)
        fustion_features = self.modality_fusion_layer(backbone_feats)        
        
        encoded_goal = self.goal_encoder(local_goal)

        fsn_global_path_feats = torch.cat([fustion_features, encoded_goal], dim=-1)
        fsn_global_path = self.global_path_predictor(fsn_global_path_feats)

        encoded_global_path = self.global_path_encoder(fsn_global_path)

        # second_layer_features = torch.cat([final_pcl_feat,final_img_feat], dim=-1)
        # global_path_encoding = self.global_path_fusion(second_layer_features)
        

        final_features_concat = torch.cat([fustion_features, encoded_global_path], dim=-1).unsqueeze(0)
        
        final_feat = self.transformer(final_features_concat)        

        final_feat = final_feat.squeeze(0)
        
        fustion_perception_path = self.joint_perception_path_feautres(final_feat)

        fusion_vel = self.predict_vel(fustion_perception_path)

        return fsn_global_path, fusion_vel,  img_path, img_vel, pcl_path, pcl_vel
