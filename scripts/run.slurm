#!/bin/bash 
#SBATCH --partition=gpuq 
#SBATCH --qos=gpu 
#SBATCH --job-name=gpu_basics 
#SBATCH --output=gpu_basics.%j.out 
#SBATCH --error=gpu_basics.%j.out 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1g.10gb:1 
#SBATCH --mem=16GB 
#SBATCH --export=ALL 
#SBATCH --time=0-01:00:00 
#
set echo 
umask 0022 
# to see ID and state of GPUs assigned
nvidia-smi 

## Load the necessary modules

module load gnu10

module load python

module load cuda/10.2

python -m venv mlp

source activate ~/mlp/bin/activate

pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

pip install torch-scatter

pip install coloredlogs

pip install spconv-cu102

cd /home/bpanigr/Research/fusion-network/scripts

## Execute your script
python pcl_main.py