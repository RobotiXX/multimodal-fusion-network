# from typing import List

import torch
import torch.nn as nn
from .backbone_fusion import ImageFusionModel
from .backbone import make_mlp
from .tf_img import CustomTransformerModelImage 



class ImageHeadMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()
        self.backbone = ImageFusionModel()        
        self.img_transformer = CustomTransformerModelImage()

        self.goal_encoder = make_mlp( [2, 128, 64], 'relu', False, False, 0.0)

        self.lstm_input = 36864+64
        self.hidden_state_dim = 512
        self.num_layers = 4

        self.rnn = nn.RNN(self.lstm_input, self.hidden_state_dim, self.num_layers, nonlinearity='relu',batch_first=True)

        self.after_rnn = nn.Sequential(
            nn.Linear(512,256),            
            nn.LeakyReLU()
        )

        self.predict_path = nn.Linear(256,8)

        self.prediction_encoder = nn.Sequential(
            nn.Linear(8, 256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU()
        )

        self.predict_vel = nn.Sequential(
            nn.Linear(512+128, 256),
            nn.ELU(),
            nn.Linear(256,2)
        )

    def forward(self, input, goal):
        
        h0 = torch.zeros(self.num_layers, 1, self.hidden_state_dim, device='cuda')
        
        image_features = self.backbone(input)                
        goal = self.goal_encoder(goal)

        img_feat_with_goal = torch.cat([image_features, goal],dim=-1)
        img_feat_with_goal = img_feat_with_goal.unsqueeze(0)

        rnn_out, _ = self.rnn(img_feat_with_goal, h0)
        
        rnn_out = rnn_out.squeeze(0)
       
        final_feat = self.after_rnn(rnn_out)
        
        prediction_path = self.predict_path(final_feat)

        encoded_prediction = self.prediction_encoder(prediction_path)

        tf_input = torch.cat([rnn_out, encoded_prediction],dim=-1).unsqueeze(0)
        
        tf_out = self.img_transformer(tf_input)

        tf_out = tf_out.squeeze(0)

        predicted_vel = self.predict_vel(tf_out)
          
        return rnn_out, prediction_path, predicted_vel




