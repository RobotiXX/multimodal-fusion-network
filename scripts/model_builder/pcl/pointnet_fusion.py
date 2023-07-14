
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
# from torch.autograd import Variabl
# import numpy as np
import torch.nn.functional as F
from .pointnet import PointNetfeat


class PointNetDenseFusionModel(nn.Module):
    def __init__(self, k = 64, feature_transform=False):
        super(PointNetDenseFusionModel, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 12, 1)
        self.conv2 = torch.nn.Conv1d(12, 10, 1)
        self.conv3 = torch.nn.Conv1d(10, 8, 1)
        self.conv4 = torch.nn.Conv1d(8, 6, 1)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.mxpool_cv2 = nn.MaxPool1d(2, 2)
        # self.mxpool_cv3 = nn.MaxPool1d(2, 2)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.conv1(x))
  
        cv2 = F.relu(self.conv2(x))

        cv3 = F.relu(self.conv3(cv2))

        x = self.conv4(cv3)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.contiguous().view(batchsize, -1)
        cv2 = cv2.contiguous().view(batchsize, -1)
        cv3 = cv3.contiguous().view(batchsize, -1)

        return x, (cv2, cv3)
