import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class FeatureEncoder(nn.Module):
    def __init__(self, feat_dim, grid_size, output_feat_dim):        
        super(FeatureEncoder, self).__init__()

        self.feat_encoder = nn.Sequential(
            nn.BatchNorm1d(feat_dim),

            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, output_feat_dim)
        )

    def forward(self, point_feat, point_index):
        device =  cur_dev = point_feat[0].get_device()
        