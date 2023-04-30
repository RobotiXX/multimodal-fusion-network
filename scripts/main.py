import torch
from data_builder.indexer import IndexDataset
from data_builder.transformer import ApplyTransformation
from model_builder.image.net import BCModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


root = '/Users/bhabaranjanpanigrahi/Research/Code/fusion-network/recorded-data/136021.bag'

indexer = IndexDataset(root)
train_dataset = ApplyTransformation(indexer)
train_loader = DataLoader(train_dataset, batch_size=16, drop_last=True)

running_loss = []
loss = torch.nn.MSELoss()
model = BCModel().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for index, (stacked_images, local_goal, prev_cmd_vel, gt_cmd_vel) in enumerate(train_loader):
        
        stacked_images = stacked_images.to(device)
        local_goal= local_goal.to(device)
        prev_cmd_vel= prev_cmd_vel.to(device)
        gt_cmd_vel=gt_cmd_vel.to(device)
        # print(f"{gt_cmd_vel.shape = }")

        pred_cmd_vel = model(stacked_images, local_goal, prev_cmd_vel)
        # print(f"{pred_cmd_vel.shape = }")
        error  = loss(pred_cmd_vel, gt_cmd_vel)
        optim.zero_grad()
        error.backward()
        optim.step()

        print(f'step is: {index} and error is: {error.item()}')

        running_loss.append(error.item())

    print(f'epoch is: {epoch} and error is: {sum(running_loss)/len(running_loss)}')

plt.plot(range(10),running_loss[:10])