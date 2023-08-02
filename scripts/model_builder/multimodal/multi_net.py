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
        self.image_weights = torch_load_weights('/home/bpanigr/Workspace/rnn_gw_img_way_pts_model_at_70.pth')
        
        self.image.load_state_dict(self.image_weights, strict=False)
        self.pcl.load_state_dict(self.pcl_weights, strict=False)

        set_trainable_false(self.image)
        set_trainable_false(self.pcl)

        self.raw_features = 1024+512
        self.hidden_state_dim = 512
        self.num_layers = 12
        self.rnn_raw_features = nn.RNN(self.raw_features, self.hidden_state_dim, self.num_layers, nonlinearity='relu',batch_first=True)

        self.final_fusion = 512+512+256
        self.final_fusion_num_layers = 24
        self.final_fusion_layer = nn.RNN(self.final_fusion, self.hidden_state_dim, self.final_fusion_num_layers, nonlinearity='relu',batch_first=True)


        self.fc = nn.Sequential(
            nn.Linear(512,512),
            nn.LeakyReLU()
        )

        self.predict = nn.Linear(512,2)

    def forward(self, stacked_images, pcl, local_goal):
        
        h1 = torch.zeros(self.num_layers, 1, self.hidden_state_dim, device='cuda')
        h2 = torch.zeros(self.final_fusion_num_layers, 1, self.hidden_state_dim, device='cuda')

        rnn_img_out, final_img_feat = self.image(stacked_images, local_goal)
        rnn_pcl_out, final_pcl_feat = self.pcl(pcl, local_goal)

        backbone_feats = torch.cat([rnn_pcl_out, rnn_img_out], dim=-1).unsqueeze(0)                
        fustion_features, _ = self.rnn_raw_features(backbone_feats, h1)
        fustion_features = fustion_features.squeeze(0)
        

        second_layer_features = torch.cat([fustion_features,final_pcl_feat, final_img_feat], dim=-1).unsqueeze(0)
        final_fusion_features, _ = self.final_fusion_layer(second_layer_features, h2)
        final_fusion_features = final_fusion_features.squeeze(0)

        prediction = self.predict(self.fc(final_fusion_features))

        return prediction
