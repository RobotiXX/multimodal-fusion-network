from typing import List

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .backbone import get_backbone
from collections import OrderedDict


class ImageFusionModel(nn.Module):
    def __init__(
        self,
        output_layers = [5,6,7],
        backbone: str = "resnet18",
        n_frames: int = 1,
        n_channels: int = 3,
        act: str = 'relu',
        l_act: bool = False, bn: bool = False, dropout: float = 0.0,

    ):

        super().__init__()
        self.output_layers = output_layers
        # print(backbone,n_frames,n_channels)
        self.backbone = get_backbone(backbone, n_frames, n_channels)
    #     print(self.backbone)
        self.max_pool = nn.MaxPool2d(kernel_size= 5, stride=2, padding=(2,2))
    #     self.selected_out = OrderedDict()
    #     self.fhooks = []

    #     for i, l in enumerate(list(self.backbone._modules.keys())):
    #         if i in self.output_layers:
    #             self.fhooks.append(getattr(self.backbone,l).register_forward_hook(self.forward_hook(l)))
            

    # def forward_hook(self, layer_name):
    #     def hook(module, input, output):
    #         self.selected_out[layer_name] = output
    #     return hook
    
    def forward(self, stacked_images):

        batchsize = stacked_images.size()[0]
        # print(f"{stacked_images.shape}")
        # image features in shape (B, 512)
        imgs = self.max_pool( self.backbone(stacked_images))
        

        linear_img_feat = imgs.contiguous().view(batchsize, -1)        

        # print(linear_img_feat.shape)
        # print(f'{pooled_features_linear.shape}')

        return linear_img_feat


# ImageFusionModel()