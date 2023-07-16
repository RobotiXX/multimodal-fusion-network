# from typing import List

import torch
import torch.nn as nn




class PclMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        self.common = nn.Sequential(
            nn.Linear(132000+128,128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,32),
            # nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.linear_vel = nn.Linear(32,1)
        self.angular_vel = nn.Linear(32,1)

    def forward(self, input):
        
        feat_shared = self.common(input)

        x = self.linear_vel(feat_shared)
        y = self.angular_vel(feat_shared)
          
        return x, y




