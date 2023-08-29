"""Different conv architectures to be used as backbone network"""

from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torchvision import models


def make_mlp(
    dims: List, act: str, l_act: bool = False, bn: bool = True, dropout: float = 0.0
):
    """Create a simple MLP with batch-norm and dropout

    Args:
        dims: (List) a list containing the dimensions of MLP
        act: (str) activation function to be used. Valid activations are [relu, tanh, sigmoid]
        l_act: (bool) whether to use activation after the last linear layer
        bn: (bool) use batch-norm or not. Default is True
        dropout: (float) dropout percentage
    """
    layers = []
    activation = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
    }[act.lower()]

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim, bias=not bn))
        if i != (len(dims) - 2):
            if bn:
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(activation)

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

    if l_act:
        layers.append(activation)

    return nn.Sequential(*layers)


def get_backbone(
    arch: str = "resnet18",
    n_frames: int = 4,
    n_channels: int = 3,
):
    """Retrieve backbone model"""
    if "resnet" in arch:
        return _get_resnet(n_frames, arch, n_channels)
    elif "mobilenet" in arch:
        return _get_mobilenet(n_frames, arch, n_channels)


def _get_resnet(
    n_frames: int = 4,
    arch: str = "resnet18",
    n_channels: int = 3,
):
    """Get different versions of ResNet and change input/output dimensions"""
    weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
    }.get(arch.lower(), models.ResNet18_Weights.DEFAULT)

    model = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }.get(arch.lower(), models.resnet18)()

    model.conv1 = nn.Conv2d(
        in_channels=n_frames * n_channels, out_channels=model.conv1.out_channels, kernel_size=model.conv1.kernel_size
    )

    if model.fc.in_features != 512:
        model.fc = nn.Linear(model.fc.in_features, 512)
    else:
        model.fc = nn.Identity()

    return model


def _get_mobilenet(
    n_frames: int = 4,
    arch: str = "resnet18",
    pretrained: bool = False,
    n_channels: int = 3,
):
    """Get different versions of mobilenet and change input/output dimensions
    Requires torchvision version 0.9.1
    """
    model = {
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "mobilenet_v2": models.mobilenet_v2,
    }.get(arch.lower(), models.mobilenet_v3_small)(pretrained=pretrained)

    model.features[0][0] = nn.Conv2d(
        in_channels=n_frames * n_channels,
        out_channels=model.features[0][0].out_channels,

    )

    if "v2" in arch.lower():
        model.classifier = nn.Linear(model.classifier[1].in_features, 512)
    else:
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 512)

    return model


def get_unet(
    model_dir: str,
    inter_repr: bool = True,
    n_frames: int = 4,
    gamma: int = 2,
    b: int = 1,
    n_channels: int = 3,
):
    """Get UNet"""
    model = UNet(gamma=gamma, b=b, inter_repr=inter_repr)
    state_dict = torch.load(Path(model_dir).resolve())
    model.load_state_dict(state_dict, strict=False)

    entry_block = EfficientConvBlock(
        in_channels=n_frames * n_channels, out_ch=3, gamma=gamma, b=b
    )

    return nn.Sequential(entry_block, model)


if __name__ == "__main__":
    # print(_get_resnet(arch='resnet18', pretrained=True))
    # print(_get_resnet(arch='resnet18')(torch.rand(1, 12, 256, 256)).shape)
    # print(_get_mobilenet(arch='mobilenet_v2', pretrained=False))
    # print(_get_mobilenet(arch='mobilenet_v2')(torch.rand(1, 12, 256, 256)).shape)
    # model = make_mlp([512, 512, 512,1], act='relu', bn=True, l_act=True)
    # print(model)
    pass
