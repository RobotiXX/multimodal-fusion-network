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

        self.lstm_input = 158400+64
        self.hidden_state_dim = 1024
        self.num_layers = 2

        self.lstm = nn.LSTM(self.lstm_input, self.hidden_state_dim, self.num_layers, batch_first=True)

        self.after_lstm = nn.Sequential(
            nn.Linear(1024,512),
            nn.ELU()                              
        )        

        self.predict = nn.Linear(512,22)            

        self.goal_encoder = make_mlp( [2, 128, 64], 'relu', False, False, 0.0)
                

    def forward(self, input, goal):
        
        batch_size = input.size()[0]

        h0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim, device='cuda')
        c0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim,device='cuda')

        point_cloud_feat = self.backbone_pcl(input.float())        
        goal = self.goal_encoder(goal)            
        
        pcl_goal_concat = torch.cat([point_cloud_feat, goal],dim=-1)
        
        pcl_goal_concat = pcl_goal_concat.unsqueeze(0)

        lstm_out, (hn, cn) = self.lstm(pcl_goal_concat, (h0,c0))

        print(f'lstm output: {lstm_out.shape}')

        lstm_out = lstm_out.squeeze(0)

        final_feat = self.after_lstm(lstm_out)
        
        prediction = self.predict(final_feat)

        return prediction



