from typing import List

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .backbone import get_backbone
from collections import OrderedDict


class ImageFusionModel(nn.Module):
    def __init__(
        self,
        output_layers = [5,6],
        backbone: str = "resnet18",
        n_frames: int = 4,
        n_channels: int = 3,
        act: str = 'relu',
        l_act: bool = False, bn: bool = False, dropout: float = 0.0,

    ):

        super().__init__()
        self.output_layers = output_layers
        self.backbone = get_backbone(backbone, n_frames, n_channels)
        self.selected_out = OrderedDict()
        self.fhooks = []

        for i, l in enumerate(list(self.backbone._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.backbone,l).register_forward_hook(self.forward_hook(l)))
            

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
    
    def forward(self, stacked_images):
        # print(f"{img.stacked_images = }")
        # image features in shape (B, 512)
        imgs = self.backbone(stacked_images)

        features = torch.cat([imgs], dim=-1)
        # return the action in shape (B, 2)
        return features, self.selected_out
