# from typing import List

import torch
import torch.nn as nn




class FusionMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        self.linear1 = nn.Linear(745471, 512)
        self.linear2 = nn.Linear(490581,256)
        self.linear3 = nn.Linear(124344,256)
        self.linear4 = nn.Linear(4*128, 1024)
        self.linear5 = nn.Linear(1024,1)

        self.angular = nn.Sequential(
            nn.Linear(1024,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        

        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.bn4 = nn.BatchNorm1d(1024)
        # self.bn5 = nn.BatchNorm1d(32)
        # self.bn6 = nn.BatchNorm1d(16)

        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.act3 = nn.LeakyReLU()
        self.act4 = nn.ReLU()
        self.act5 = nn.ReLU()
        self.act6 = nn.ReLU()


    def forward(self, input_l1, input_l2, input_l3, goal, prev_cmd_vel):
        
        x = self.linear1(input_l1)
        x = self.act1(x)        
       

        x = torch.cat([input_l2,x], dim=-1)
        x = self.linear2(x)
        x = self.act2(x)          

        x = torch.cat([input_l3,x], dim=-1)
        x = self.linear3(x)
        x = self.act3(x)   
      
        # print(x.shape,goal.shape, prev_cmd_vel.shape)
        x = torch.cat([x, goal, prev_cmd_vel], dim=-1)
        x = self.linear4(x)
        x_shared = self.act4(x)  

        x = self.linear5(x_shared)

        y = self.angular(x_shared)
          
        return x, y




