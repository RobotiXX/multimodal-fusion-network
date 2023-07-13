from comet_ml import Experiment
import torch
import os
import numpy as np
from random import shuffle

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
    # project_name= "multimodal-net-with-rnn",
    project_name="kk",
    workspace="bhabaranjan",
)

coloredlogs.install()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')

min_val_error = 100000

val_dict = {}

def get_data_loader(input_file_path, read_type, batch_size):
    logging.info(f'Reading {read_type} file from path {input_file_path}')
    indexer = IndexDataset(input_file_path)
    transformer = ApplyTransformation(indexer)
    data_loader = DataLoader(transformer, batch_size = batch_size, drop_last=True)
    return data_loader

def get_loss(loss_fn, lin_vel, angular_vel, gt_lin, gt_angular, data_src):
    lin_error =  loss_fn(lin_vel, gt_lin) 
    anglr_error = loss_fn(angular_vel, gt_angular)
    error = lin_error + anglr_error
    # print(f'linear error: {lin_error}')
    # print(f'angular error: {anglr_error}')

    experiment.log_metric(name = str('line_error_'+data_src), value=lin_error.item())
    experiment.log_metric(name = str('anglr_error_'+data_src), value=anglr_error.item())
    return error


def run_validation(val_files, model, batch_size, epoch, optim):
       print("Running Validation..\n")
       running_error = []
       loss = torch.nn.MSELoss()
       with torch.no_grad():
        for val_file in val_files:        
            val_loader = None
            if val_file not in val_dict:
                val_loader = get_data_loader( val_file, 'validation', batch_size = batch_size )
                val_dict[val_file] = val_loader
            else:
                val_loader = val_dict[val_file]

            per_file_loss_fusion  = []
            per_file_loss_ǐmage = []
            per_file_loss_pcl = []
            per_file_total_loss = []
            for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in tqdm(enumerate(val_loader)):
                stacked_images = stacked_images.to(device)
                pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                prev_cmd_vel= prev_cmd_vel.to(device)
                gt_cmd_vel= gt_cmd_vel.to(device)
                
                fsn_lin, fsn_anglr, img_lin, img_anglr, pcl_lin, pcl_anglr = model(stacked_images, pcl, local_goal, prev_cmd_vel)
                

                gt_x = torch.unsqueeze(gt_cmd_vel[:,0],1)
                gt_y = torch.unsqueeze(gt_cmd_vel[:,1],1)

                # print(fsn_lin/10)
                # print(fsn_anglr/100)
 
                # print(gt_x)
                # print(gt_y)
                
                error_fusion = get_loss(loss, fsn_lin/100, fsn_anglr/1000, gt_x, gt_y,'fusion')
                error_img = get_loss(loss, img_lin/100, img_anglr/1000, gt_x, gt_y, 'img')
                error_pcl = get_loss(loss, pcl_lin/100, pcl_anglr/1000, gt_x, gt_y, 'pcl')
                
                error_total = error_fusion + ( 0.2 * error_img) + (0.8 * error_pcl)

                per_file_loss_fusion.append(error_fusion.item())
                per_file_loss_ǐmage.append(error_img.item())
                per_file_loss_pcl.append(error_pcl.item())                
                per_file_total_loss.append(error_total.item())
                
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_img'), value=np.average(per_file_loss_ǐmage), epoch = epoch + 1)
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_pcl'), value=np.average(per_file_loss_pcl), epoch = epoch + 1)
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_fusion'), value=np.average(per_file_loss_fusion), epoch = epoch + 1)

            running_error.append(np.average(per_file_loss_fusion))
            print(f'================== epoch is: {epoch} and error is: {np.average(per_file_loss_fusion)}==================\n')
        avg_loss_on_validation = np.average(running_error)
        
        if (epoch+1) % 10 == 0 and epoch!=0:
            min_val_error = avg_loss_on_validation
            print(f"saving model weights at validation error {min_val_error}")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, f'model_at_{epoch+1}.pth')

        print(f'=========================> Average Validation error for fusion is:   {avg_loss_on_validation} \n')
        return avg_loss_on_validation
            


def run_training(train_files, val_dirs, batch_size, num_epochs):
    loss = torch.nn.MSELoss()
    model = BcFusionModel()
    ckpt = torch.load("/home/ranjan/Workspace/my_works/fusion-network/scripts/prev_model_at_30.pth")
    model.load_state_dict(ckpt['model_state_dict'])
    # model.eval()


    optim = torch.optim.Adam(model.parameters(), lr = 0.01) 

    model.to(device)
    run_validation(val_dirs, model, batch_size, 0, optim)
    
    model.train()
    val_error_at_epoch = []
    
    # scheduler = MultiStepLR(optim, milestones=[2,12,27,42], gamma=0.1)
    epoch_loss = []
    data_dict = {}
    # for epoch in range(num_epochs):
    #     num_files = 0
        # lr = scheduler.get_last_lr()
        # experiment.log_metric( name = "Learning Rate Decay", value = lr, epoch= epoch+1)
        # running_loss = []
        # shuffle(train_files)
        # for train_file in train_files:        
        #     train_loader = None 
        #     print(train_file)
        #     if train_file not in data_dict:
        #         train_loader = get_data_loader( train_file, 'train', batch_size = batch_size )   
        #         data_dict[train_file] = train_loader
        #     else:
        #         train_loader = data_dict[train_file]

        #     num_files += 1
        #     per_file_loss_fusion = [] 
        #     per_file_loss_ǐmage = [] 
        #     per_file_loss_pcl = [] 
        #     per_file_total_loss = []
        #     for index, (stacked_images, pcl ,local_goal, prev_cmd_vel, gt_cmd_vel) in enumerate(train_loader):

        #         # print(f'gt_cmd: {gt_cmd_vel}')
        #         # print(f'prev_cmd_vel:{prev_cmd_vel}')
                
        #         stacked_images = stacked_images.to(device)
        #         pcl = pcl.to(device)
        #         local_goal= local_goal.to(device)
        #         prev_cmd_vel= prev_cmd_vel.to(device)
        #         gt_cmd_vel= gt_cmd_vel.to(device)
        #         # print(f"{gt_cmd_vel.shape = }")
        #         optim.zero_grad()
                
        #         fsn_lin, fsn_anglr, img_lin, img_anglr, pcl_lin, pcl_anglr = model(stacked_images, pcl, local_goal, prev_cmd_vel)
                
        #         gt_x = torch.unsqueeze(gt_cmd_vel[:,0],1)
        #         gt_y = torch.unsqueeze(gt_cmd_vel[:,1],1)

        #         # print(fsn_lin)
        #         # print(fsn_anglr)

        #         # print(gt_x)
        #         # print(gt_y)
                
        #         error_fusion = get_loss(loss, fsn_lin, fsn_anglr, gt_x, gt_y,'train_fusion')
        #         error_img = get_loss(loss, img_lin, img_anglr, gt_x, gt_y, 'train_img')
        #         error_pcl = get_loss(loss, pcl_lin, pcl_anglr, gt_x, gt_y,'train_pcl')
                
        #         error_total = error_fusion + error_img + error_pcl

        #         per_file_loss_fusion.append(error_fusion.item())
        #         per_file_loss_ǐmage.append(error_img.item())
        #         per_file_loss_pcl.append(error_pcl.item())                
        #         per_file_total_loss.append(error_total.item())

        #         error_total.backward()
        #         optim.step()

        #         per_file_loss_fusion.append(error_fusion.item())
        #         per_file_loss_ǐmage.append(error_img.item())
        #         per_file_loss_pcl.append(error_pcl.item())
        #         per_file_total_loss.append(error_total.item())

        #         print(f'step is:   {index} and total error is:   {error_total.item()}  image: {error_img.item()}  pcl: {error_pcl.item()} fusion: {error_fusion.item()}\n')
            
        #     experiment.log_metric(name = str(train_file.split('/')[-1]+ " mod:" +'img'), value=np.average(per_file_loss_ǐmage), epoch= epoch+1)
        #     experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'pcl'), value=np.average(per_file_loss_pcl), epoch= epoch+1)
        #     experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'fusion'), value=np.average(per_file_loss_fusion), epoch= epoch+1)
        #     running_loss.append(np.average(per_file_total_loss))   
            
               
        # print(f'================== epoch is: {epoch} and error is: {np.average(running_loss)}==================\n')
        # if (epoch+1) % 2 == 0:
        #     val_error = run_validation(val_dirs, model, batch_size, epoch, optim)
        #     experiment.log_metric( name = "Avg Validation loss", value = np.average(val_error), epoch= epoch+1)
        # # val_error_at_epoch.append(val_error)
        # experiment.log_metric( name = "Avg Training loss", value = np.average(running_loss), epoch= epoch+1)
        


def main():
    train_path = "../recorded-data/train"
    # train_path = "../recorded-data/sandbox"
    train_dirs = [ os.path.join(train_path, dir) for dir in os.listdir(train_path)]
    val_dirs = [ os.path.join('../recorded-data/val', dir) for dir in os.listdir('../recorded-data/val')]
    batch_size = 12
    epochs = 250 
    run_training(train_dirs, val_dirs, batch_size, epochs)



main()



