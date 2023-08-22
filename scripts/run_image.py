from comet_ml import Experiment
import torch
import os
import numpy as np
from random import shuffle

from data_builder.indexer import IndexDataset
from data_builder.transformer import ApplyTransformation
from data_builder.gaussian_weights import get_gaussian_weights
from data_builder.cmd_scaler import transform_to_gt_scale
from model_builder.multimodal.fusion_net import BcFusionModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import coloredlogs, logging
from model_builder.image.image_head import ImageHeadMLP
import math

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, CyclicLR

# Create an experiment with your api key
experiment = Experiment(
    api_key="Ly3Tc8W7kfxPmAxsArJjX9cgo",
    # project_name= "test",
    project_name="image_only",
    workspace="bhabaranjan",
)

experiment.add_tag('rnn-image-model')

coloredlogs.install()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')

min_val_error = 100000

val_dict = {}

model_storage_path = '/scratch/bpanigr/model_weights/unimodals'

weights = get_gaussian_weights(2,1.3)
weights = weights[:,:4] 
weights = np.concatenate([weights, weights], axis=1)
weights = torch.tensor(weights)
weights = weights.to(device)


def clear_dict(dict, epoch):
    
    if epoch % 25 == 0:
        dict.clear()

    return

def get_data_loader(input_file_path, read_type, batch_size):
    logging.info(f'Reading {read_type} file from path {input_file_path}')
    indexer = IndexDataset(input_file_path)
    transformer = ApplyTransformation(indexer)
    data_loader = DataLoader(transformer, batch_size = batch_size, drop_last=False, prefetch_factor=2,num_workers=8)
    return data_loader

def get_loss_prev(loss_fn, lin_vel, angular_vel, gt_lin, gt_angular, data_src):
    lin_error =  loss_fn(lin_vel, gt_lin) 
    anglr_error = loss_fn(angular_vel, gt_angular)
    error = lin_error + anglr_error
    
    lin_err_val = lin_error.item()
    anglr_error_val = anglr_error.item()

    experiment.log_metric(name = str('line_error_'+data_src), value= lin_err_val)
    experiment.log_metric(name = str('anglr_error_'+data_src), value=anglr_error_val)


    return error

def get_loss(loss_fn, pts, gt_pts, data_src):
    error =  loss_fn(pts, gt_pts)

    if data_src == 'validation':     
        experiment.log_metric(name = str('way_pts'+data_src), value= error.item())    
    else:
        experiment.log_metric(name = str('way_pts'+data_src), value=error.item())    
    return error

def std_print(file,label, value):
    print(f'{file} {label}: {value}')

def run_validation(val_files, model, batch_size, epoch, optim):
       print("Running Validation..\n")
       running_error = []
       loss = torch.nn.L1Loss()
       model.eval()
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
            per_file_loss_pcl_cmd = []
            for index, (stacked_images, pcl ,local_goal, gt_pts, gt_cmd) in tqdm(enumerate(val_loader)):
                stacked_images = stacked_images.to(device)
                # pcl = pcl.to(device)
                local_goal= local_goal.to(device)
                
                gt_pts= gt_pts.to(device)
                gt_cmd= gt_cmd.to(device)
                
                pts, vel = model(stacked_images, local_goal)

                                
                vel = transform_to_gt_scale(vel, device)                
                gt_cmd = transform_to_gt_scale(gt_cmd, device)  

                # print('\nstart----')
                # print(pcl_lin)
                # print(pcl_anglr)
 
                # print(gt_x)
                # print(gt_y)
                # print('end\n')
                
                # error_fusion = get_loss(loss, fsn_lin, fsn_anglr, gt_x, gt_y,'fusion')
                # error_img = get_loss(loss, img_lin, img_anglr, gt_x, gt_y, 'img')
                error_pcl = get_loss(loss, pts/weights, gt_pts/weights, 'validation')
                error_cmd_vel = get_loss(loss, vel, gt_cmd,'validation')
                
                # error_total = error_fusion + ( 0.2 * error_img) + error_pcl

                # per_file_loss_fusion.append(error_fusion.item())
                # per_file_loss_ǐmage.append(error_img.item())
                per_file_loss_pcl.append(error_pcl.item())  
                per_file_loss_pcl_cmd.append(error_cmd_vel.item())                
                # per_file_total_loss.append(error_total.item())
                            
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_pcl'), value=np.average(per_file_loss_pcl), epoch = epoch + 1)
            experiment.log_metric(name = str('val_'+val_file.split('/')[-1]+'_pcl_cmd'), value=np.average(per_file_loss_pcl_cmd), epoch = epoch + 1)
            std_print(str('val_'+val_file.split('/')[-1]), 'path',np.average(per_file_loss_pcl))
            std_print(str('val_'+val_file.split('/')[-1]), 'cmd',np.average(per_file_loss_pcl_cmd))
        
            running_error.append(np.average(per_file_loss_pcl))
        
        avg_loss_on_validation = np.average(running_error)
        # print(f'epoch:------>{epoch}')
        if (epoch+1) % 10 == 0 and (epoch+1) > 20:
            print(f"saving model weights at validation error {avg_loss_on_validation}")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, f'{model_storage_path}/unimodal_img_{epoch+1}_{avg_loss_on_validation}.pth')

        print(f'=========================> Average Validation error is:   {avg_loss_on_validation} \n')
        return avg_loss_on_validation
            


def run_training(train_files, val_dirs, batch_size, num_epochs):
    loss = torch.nn.L1Loss()
    model = ImageHeadMLP()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 0.0000288)     
    
    # run_validation(val_dirs, model, batch_size, 2, optim)  
    # ckpt = torch.load('/home/ranjan/Workspace/my_works/fusion-network/scripts/pre_img_way_pts_model_at_110.pth')
    # model.load_state_dict(ckpt['model_state_dict'])
    # run_validation(val_dirs, model, batch_size, 0, optim)
    # return
    # run_validation(val_dirs, model, batch_size, 0, optim)
    
    scheduler = MultiStepLR(optim, milestones= [20,70,110], gamma=.8)

    data_dict = {}
    for epoch in range(num_epochs):
        num_files = 0
        lr = scheduler.get_last_lr()        
        experiment.log_metric( name = "Learning Rate Decay", value = lr, epoch= epoch+1)
        running_loss = []
        shuffle(train_files)        
        model.train()        
        clear_dict(data_dict, epoch)
        for train_file in train_files:        
            train_loader = None 
            print(train_file)
            if train_file not in data_dict:
                train_loader = get_data_loader( train_file, 'train', batch_size = batch_size )   
                data_dict[train_file] = train_loader
            else:
                train_loader = data_dict[train_file]

            num_files += 1
            per_file_loss_fusion = [] 
            per_file_loss_ǐmage = [] 
            per_file_loss_pcl = [] 
            per_file_loss_pcl_cmd = [] 
            per_file_total_loss = []
            for index, (stacked_images, pcl ,local_goal, gt_pts, gt_cmd) in enumerate(train_loader):

                # print(f'gt_cmd: {gt_cmd_vel}')
                # print(f'prev_cmd_vel:{prev_cmd_vel}')
                
                stacked_images = stacked_images.to(device)
                # pcl = pcl.to(device)
                local_goal= local_goal.to(device)                
                gt_pts= gt_pts.to(device)
                gt_cmd= gt_cmd.to(device)
                # print(f"{pcl.shape = }")
                optim.zero_grad()
                
                pts, cmd = model(stacked_images, local_goal)
                error_pcl = get_loss(loss, pts, gt_pts,'train_pcl')
                error_pcl_cmd = get_loss(loss, cmd, gt_cmd,'train_pcl')

                per_file_loss_pcl.append(error_pcl.item())     
                per_file_loss_pcl_cmd.append(error_pcl_cmd.item())   

                total_error = error_pcl + error_pcl_cmd
                
                total_error.backward()
                optim.step()

                print(f'step is:   {index} and total path error is :: {error_pcl.item()} total cmd_vel error is : {error_pcl_cmd.item()} \n')

            
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'pcl'), value=np.average(per_file_loss_pcl), epoch= epoch+1)
            experiment.log_metric(name = str(train_file.split('/')[-1]+" mod:" +'cmd'), value=np.average(per_file_loss_pcl_cmd), epoch= epoch+1)

            std_print(str(train_file.split('/')[-1]), 'path', np.average(per_file_loss_pcl))
            std_print(str(train_file.split('/')[-1]), 'cmd', np.average(per_file_loss_pcl_cmd))

            running_loss.append(np.average(per_file_loss_pcl))   
            
        scheduler.step()       
        print(f'================== epoch is: {epoch} and error is: {np.average(running_loss)}==================\n')

        if (epoch+1) % 2 == 0:
            val_error = run_validation(val_dirs, model, batch_size, epoch, optim)
            experiment.log_metric( name = "Avg Validation loss", value = np.average(val_error), epoch= epoch+1)
        # val_error_at_epoch.append(val_error)
        experiment.log_metric( name = "Avg Training loss", value = np.average(running_loss), epoch= epoch+1)
        


def main():
    train_path = "/scratch/bpanigr/fusion-network/recorded-data/train"
    # train_path = "../recorded-data/train"
    train_dirs = [ os.path.join(train_path, dir) for dir in os.listdir(train_path)]
    # validation_path = '../recorded-data/val'
    validation_path = '/scratch/bpanigr/fusion-network/recorded-data/val'
    val_dirs = [ os.path.join(validation_path, dir) for dir in os.listdir(validation_path)]


    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/136021_wt')
    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/138181_wt')
    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/135968_wt_at')
    # # train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/136514_sw_wt_sc')
    train_dirs.remove('/scratch/bpanigr/fusion-network/recorded-data/train/135967_at')


    batch_size = 90
    epochs = 250
    run_training(train_dirs, val_dirs, batch_size, epochs)



main()



