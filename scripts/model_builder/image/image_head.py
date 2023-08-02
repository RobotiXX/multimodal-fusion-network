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
        self.goal_encoder = make_mlp( [2, 64, 128, 64], 'relu', False, False, 0.0)

        self.lstm_input = 36864+64
        self.hidden_state_dim = 512
        self.num_layers = 4

        self.rnn = nn.RNN(self.lstm_input, self.hidden_state_dim, self.num_layers, nonlinearity='relu',batch_first=True)

        self.after_rnn = nn.Sequential(
            nn.Linear(512,256),            
            nn.LeakyReLU()
        )

        self.predict = nn.Linear(256,22)

    def forward(self, input, goal):
        
        h0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim, device='cuda')
        
        image_features = self.backbone(input)                
        goal = self.goal_encoder(goal)

        img_feat_with_goal = torch.cat([image_features, goal],dim=-1)

        
        img_feat_with_goal = img_feat_with_goal.unsqueeze(0)

        rnn_out, _ = self.rnn(img_feat_with_goal, h0)
        
        rnn_out = rnn_out.squeeze(0)
       
        final_feat = self.after_rnn(rnn_out)
        
        # prediction = self.predict(final_feat)
          
        return rnn_out, final_feat




