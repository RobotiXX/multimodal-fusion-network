from comet_ml import Experiment
import torch
import os
import numpy as np

from data_builder.indexer import IndexDataset
from data_builder.transformer import ApplyTransformation
from model_builder.multimodal.fusion_net import BcFusionModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import coloredlogs, logging

from torch.optim.lr_scheduler import MultiStepLR

# Create an experiment with your api key
experiment = Experiment(
    api_key="Ly3Tc8W7kfxPmAxsArJjX9cgo",
    project_name="multimodal-larning",
    workspace="bhabaranjan",
)

coloredlogs.install()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')


def get_data_loader(input_file_path, read_type, batch_size):
    logging.info(f'Reading {read_type} file from path {input_file_path}')
    indexer = IndexDataset(input_file_path)
    transformer = ApplyTransformation(indexer)
    data_loader = DataLoader(transformer, batch_size = batch_size)
    return data_loader


def run_validation(val_files, model, batch_size):
       print("Running Validation..\n")
       val_error = []
       loss = torch.nn.MSELoss()
       with torch.no_grad():
        for val_file in val_files:        
            val_loader = get_data_loader( val_file, 'validation', batch_size = batch_size )
            for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in tqdm(enumerate(val_loader)):
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                prev_cmd_vel= prev_cmd_vel.to(device)
                gt_cmd_vel= gt_cmd_vel.to(device)
                pred_fusion, pred_img, pred_pcl = model(stacked_images, pcl, local_goal, prev_cmd_vel)
                error_fusion = loss(pred_fusion, gt_cmd_vel)
                error_img = loss(pred_img, gt_cmd_vel)
                error_pcl = loss(pred_pcl, gt_cmd_vel)
                error_total = error_fusion + ( 0.25 * error_img) + (0.75 * error_pcl)
                error_to_number = error_total.item()
                val_error.append(error_to_number)
                experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_img'), value=error_img.item())
                experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_pcl'), value=error_pcl.item())
                experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_fusion'), value=error_fusion.item())
        running_loss = sum(val_error)/len(val_error)

        print(f'=========================> Average Validation error is:   {running_loss} \n')
        return running_loss
            


def run_training(train_files, val_dirs, batch_size, num_epochs):
    loss = torch.nn.MSELoss()
    model = BcFusionModel().to(device)
    error_at_epoch = []
    val_error_at_epoch = []
    optim = torch.optim.Adam(model.parameters(), lr=0.1) 
    scheduler = MultiStepLR(optim, milestones=[1,2,4], gamma=0.1)
    epoch_loss = []
    for epoch in range(num_epochs):
        num_files = 0
        lr = scheduler.get_last_lr()
        experiment.log_metric( name = "Learning Rate Decay", value = lr, epoch= epoch+1)
        running_loss = []
        for train_file in train_files:        
            train_loader = get_data_loader( train_file, 'train', batch_size = batch_size )   
            num_files += 1
            per_file_loss_fusion = [] 
            per_file_loss_ǐmage = [] 
            per_file_loss_pcl = [] 
            per_file_total_loss = []
            for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in enumerate(train_loader):
                
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                prev_cmd_vel= prev_cmd_vel.to(device)
                gt_cmd_vel= gt_cmd_vel.to(device)
                # print(f"{gt_cmd_vel.shape = }")

                pred_fusion, pred_img, pred_pcl  = model(stacked_images, pcl, local_goal, prev_cmd_vel)
                # print(f"{pred_cmd_vel.shape = }")
                # print(pred_fusion, gt_cmd_vel)
                error_fusion = loss(pred_fusion, gt_cmd_vel)
                error_img = loss(pred_img, gt_cmd_vel)
                error_pcl = loss(pred_pcl, gt_cmd_vel)
                error_total = error_fusion + ( 0.25 * error_img) + (0.75 * error_pcl)
                

                optim.zero_grad()
                error_total.backward()
                optim.step()

                per_file_loss_fusion.append(error_fusion.item())
                per_file_loss_ǐmage.append(error_img.item())
                per_file_loss_pcl.append(error_pcl.item())
                per_file_total_loss.append(error_total.item())

                print(f'step is:   {index} and total error is:   {error_total.item()}  image: {error_img.item()}  pcl: {error_pcl.item()} fusion: {error_fusion.item()}\n')
            
            experiment.log_metric(name = str(train_file.split('/')[-1]+ " mod:" +'img'), value=np.average(per_file_loss_ǐmage), epoch= epoch+1)
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'pcl'), value=np.average(per_file_loss_pcl), epoch= epoch+1)
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'fusion'), value=np.average(per_file_loss_fusion), epoch= epoch+1)
            running_loss.append(np.average(per_file_total_loss))   
            
            if num_files%6 == 0:  
                print("After trained on 6 files..")              
                run_validation(val_dirs, model, batch_size)
        
        epoch_loss.append(np.average(running_loss))
        scheduler.step()
        avg_error_at_epoch = sum(running_loss)/len(running_loss)
        error_at_epoch.append(avg_error_at_epoch)
        print(f'================== epoch is: {epoch} and error is: {error_at_epoch}==================\n')
        val_error = run_validation(val_dirs, model, batch_size)
        val_error_at_epoch.append(val_error)
        experiment.log_metric( name = "Avg Training loss", value = np.average(epoch_loss), epoch= epoch+1)
        experiment.log_metric( name = "Avg Validation loss", value = val_error, epoch= epoch+1)
        
    torch.save(model.state_dict(), "saved_fusion_model.pth")



def main():
    train_dirs = [ os.path.join('../recorded-data/train', dir) for dir in os.listdir('../recorded-data/train')]
    val_dirs = [ os.path.join('../recorded-data/val', dir) for dir in os.listdir('../recorded-data/val')]
    batch_size = 16
    epochs = 45
    run_training(train_dirs, val_dirs, batch_size, epochs)



main()