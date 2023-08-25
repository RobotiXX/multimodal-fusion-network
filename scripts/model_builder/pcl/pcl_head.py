# from typing import List

import torch
import torch.nn as nn
from ..image.backbone import make_mlp
from .pointnet_backbone import PclBackbone
from .tf_pcl import CustomTransformerModelPcl



class PclMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        self.backbone_pcl = PclBackbone().float()
        self.pcl_transformer = CustomTransformerModelPcl()
        self.goal_encoder = make_mlp( [2, 128, 64], 'relu', False, False, 0.0)

        self.lstm_input = 158400+64
        self.hidden_state_dim = 1024
        self.num_layers = 4

        self.rnn = nn.RNN(self.lstm_input, self.hidden_state_dim, self.num_layers, nonlinearity='relu',batch_first=True)

        self.after_rnn = nn.Sequential(
            nn.Linear(1024,512),
            nn.ELU()                              
        )        

        self.predict_path = nn.Linear(512,8)            

        self.predict_path_encoder = nn.Sequential(
            nn.Linear(8, 256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU()                            
        )        

        self.predict_vel = nn.Sequential(
            nn.Linear(1024+128, 512),
            nn.ELU(),
            nn.Linear(512,2)
        )
        
                

    def forward(self, input, goal):

        h0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim, device='cuda')
        # c0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim,device='cuda')

        point_cloud_feat = self.backbone_pcl(input.float())        
        goal = self.goal_encoder(goal)            
        
        pcl_goal_concat = torch.cat([point_cloud_feat, goal],dim=-1)
        
        pcl_goal_concat = pcl_goal_concat.unsqueeze(0)

        rnn_out, _ = self.rnn(pcl_goal_concat, h0)

        rnn_out = rnn_out.squeeze(0)

        final_feat = self.after_rnn(rnn_out)

        prediction_path = self.predict_path(final_feat)                

        encoded_path = self.predict_path_encoder(prediction_path)

        tf_input = torch.cat([rnn_out, encoded_path],dim=-1).unsqueeze(0)
        
        tf_out = self.pcl_transformer(tf_input)

        tf_out = tf_out.squeeze(0)

        predicted_vel = self.predict_vel(tf_out)

        return rnn_out, prediction_path, predicted_vel



