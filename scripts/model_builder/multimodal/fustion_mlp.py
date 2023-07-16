# from typing import List

import torch
import torch.nn as nn




class FusionMLP(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()


        self.linear1 = nn.Linear(60000+41472, 512)
        self.linear2 = nn.Linear(48000+20736+512,256)
        self.linear3 = nn.Linear(36000+25088+256,256)
        self.linear4 = nn.Linear(3*128, 512)
        self.linear5 = nn.Linear(512,1)

        self.angular = nn.Sequential(
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        

        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.act3 = nn.LeakyReLU()
        self.act4 = nn.ReLU()


    def forward(self, input_l1, input_l2, input_l3, goal):
        
        x = self.linear1(input_l1)
        x = self.act1(x)        
       

        x = torch.cat([input_l2,x], dim=-1)
        x = self.linear2(x)
        x = self.act2(x)          

        x = torch.cat([input_l3,x], dim=-1)
        x = self.linear3(x)
        x = self.act3(x)   
      
        # print(x.shape,goal.shape, prev_cmd_vel.shape)
        x = torch.cat([x, goal], dim=-1)
        x = self.linear4(x)
        x_shared = self.act4(x)  

        x = self.linear5(x_shared)

        y = self.angular(x_shared)
          
        return x, y




