import torch
from data_builder.indexer import IndexDataset
from data_builder.transformer import ApplyTransformation
from model_builder.multimodal.fusion_net import FusionModel
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"



root = '/home/ranjan/Workspace/fusion-network/recorded-data/train/136021'

indexer = IndexDataset(root)
train_dataset = ApplyTransformation(indexer)
batch_size=16
train_loader = DataLoader(train_dataset,batch_size=16, drop_last=True)

running_loss = []
loss = torch.nn.MSELoss()
model = FusionModel().to(device)
print(model)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for index, (stacked_images, points_clouds, local_goal, prev_cmd_vel, gt_cmd_vel) in enumerate(train_loader):
        
        stacked_images = stacked_images.to(device)
        points_clouds = points_clouds.to(device)
        
        local_goal= local_goal.to(device)
        prev_cmd_vel= prev_cmd_vel.to(device)
        gt_cmd_vel= gt_cmd_vel.to(device)
        # print(f"{gt_cmd_vel.shape = }")

        pred_cmd_vel, img_rep, ptcld_rep = model(stacked_images, points_clouds, local_goal, prev_cmd_vel)
        # print(itm)
        for key, val in img_rep.items():
            print(key, "  -  ", val.shape)
        # print(f"{pred_cmd_vel.shape = }")
        # error  = loss(pred_cmd_vel, gt_cmd_vel)
        # optim.zero_grad()
        # error.backward()
        # optim.step()
        print(ptcld_rep[0].shape, ptcld_rep[1].shape)

        # print(f'step is: {index} and error is: {error.item()}')

        # running_loss.append(error.item())

    # print(f'epoch is: {epoch} and error is: {sum(running_loss)/len(running_loss)}')

# plt.plot(range(10),running_loss[:10])