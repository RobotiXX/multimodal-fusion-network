import torch
import os

from data_builder.indexer import IndexDataset
from data_builder.transformer import ApplyTransformation
from model_builder.multimodal.net import BCModelPcl
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import coloredlogs, logging


coloredlogs.install()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')


def get_data_loader(input_file_path, read_type, batch_size):
    logging.info(f'Reading {read_type} file from path {input_file_path}')
    indexer = IndexDataset(input_file_path)
    transformer = ApplyTransformation(indexer)
    data_loader = DataLoader(transformer, batch_size = batch_size, drop_last=True)
    return data_loader



# root = '/Users/bhabaranjanpanigrahi/Research/Code/fusion-network/recorded-data/136021.bag'

root_val = '/home/ranjan/Workspace/fusion-network/recorded-data/val/138139'

val_indexer = IndexDataset(root_val)
val_dataset = ApplyTransformation(val_indexer)
val_loader = DataLoader(val_dataset, batch_size=24, drop_last=True)


def run_validation(val_loader, model):
       print("Running Validation..\n")
       val_error = []
       loss = torch.nn.MSELoss()
       with torch.no_grad():
        for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in tqdm(enumerate(val_loader)):
            stacked_images = stacked_images.to(device)
            pcl = pcl.to(device)
            local_goal= local_goal.to(device)
            prev_cmd_vel= prev_cmd_vel.to(device)
            gt_cmd_vel= gt_cmd_vel.to(device)
            pred_cmd_vel = model(pcl, stacked_images, local_goal, prev_cmd_vel)
            error = loss(pred_cmd_vel, gt_cmd_vel)
            error_to_number = error.item()
            val_error.append(error_to_number)

        running_loss = sum(val_error)/len(val_error)

        print(f' =======> Average Validation error is:   {running_loss} \n')
        return running_loss
            


def run_training(train_files, batch_size, num_epochs):
    loss = torch.nn.MSELoss()
    model = BCModelPcl().to(device)
    error_at_epoch = []
    val_error_at_epoch = []
    optim = torch.optim.Adam(model.parameters(), lr=0.0001) 
    
    for epoch in range(num_epochs):   
        running_loss = []
        for train_file in train_files:        
            train_loader = get_data_loader( train_file, 'train', batch_size = batch_size )            
            for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in enumerate(train_loader):
                
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                prev_cmd_vel= prev_cmd_vel.to(device)
                gt_cmd_vel= gt_cmd_vel.to(device)
                # print(f"{gt_cmd_vel.shape = }")

                pred_cmd_vel = model(pcl, stacked_images, local_goal, prev_cmd_vel)
                # print(f"{pred_cmd_vel.shape = }")
                error = loss(pred_cmd_vel, gt_cmd_vel)
                optim.zero_grad()
                error.backward()
                optim.step()

                print(f'step is:   {index} and error is:   {error.item()} \n')
                running_loss.append(error.item())

        avg_error_at_epoch = sum(running_loss)/len(running_loss)
        error_at_epoch.append(avg_error_at_epoch)
        print(f'================== epoch is: {epoch} and error is: {error_at_epoch}==================\n')
        val_error_at_epoch.append(run_validation(val_loader, model))

    torch.save(model.state_dict(), "saved_model.pth")
    plt.plot(range(len(error_at_epoch)),error_at_epoch,'-',color='g', label= 'Training')
    plt.plot(range(len(val_error_at_epoch)),val_error_at_epoch,'--',color='r', label='Testing')
    plt.legend(loc='upper left')
    plt.show()



def main():
    train_dirs = [ os.path.join('../recorded-data/train', dir) for dir in os.listdir('../recorded-data/train')]
    batch_size = 24
    epochs = 10
    run_training(train_dirs, batch_size, epochs)



main()