# from typing import List

import torch
import torch.nn as nn




class PclBackbone(nn.Module):
    def __init__(
        self,        
    ):

        super().__init__()

        
        self.common = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=5, stride=1),
            nn.LeakyReLU(),   
            nn.MaxPool3d(3, stride=(2,2,1))                   
            )

        self.ft1 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=24, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(3, stride=(2,2,1))                   
        )

        self.ft2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=36, kernel_size=5, stride=1),
            nn.LeakyReLU(),   
            nn.MaxPool3d(3, stride=(2,2,1))                            
        )

        self.ft3 = nn.Sequential(
            nn.Conv3d(in_channels=36, out_channels=48, kernel_size=1, stride=1),
            nn.LeakyReLU(),    
        )                
        

    def forward(self, input):
             
        batchsize = input.size()[0]
        # print(input.size())
        feat_shared = self.common(input)

        feat_l1 = self.ft1(feat_shared)
        feat_l2 = self.ft2(feat_l1)
        feat_l3 = self.ft3(feat_l2)

        # feat_l1 = feat_l1.contiguous().view(batchsize, -1)
        # feat_l2 = feat_l2.contiguous().view(batchsize, -1)
        feat_l3 = feat_l3.contiguous().view(batchsize, -1)        
        
        return feat_l3
