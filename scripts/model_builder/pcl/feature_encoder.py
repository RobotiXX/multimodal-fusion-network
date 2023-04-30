import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class FeatureEncoder(nn.Module):
    def __init__(self , grid_size, output_feat_dim = 64, feat_dim = 9):        
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
        device = point_feat[0].get_device()
        
        # concat everything
        cat_point_index = []
        
        for batch_i in  range(len(point_index)):
            cat_point_index.append(F.pad(point_index[batch_i], (1,0), mode='constant', value=batch_i))
        
        point_feat = torch.cat(point_feat, dim=0)
        cat_point_index = torch.cat(cat_point_index, dim=0)        

        # shuffle data
        num_points = cat_point_index.shape[0]
        
        shuffled_ind = torch.randperm(num_points, device= device)
        point_feat = point_feat[shuffled_ind, :]
        cat_point_index = cat_point_index[shuffled_ind, :]

        # unique xy grid index
        unq_index, unq_inv, unq_cnt = torch.unique(cat_point_index, return_inverse=True, return_counts=True, dim=0)
        unq_index = unq_index.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(point_feat)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        return unq_index, pooled_data