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

module load spack

spack install gcc@10.0.0

spack load gcc@10.0.0

module load cuda/11.3

python -m venv mlp

source activate ~/mlp/bin/activate

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install torch-scatter

pip install coloredlogs

pip install spconv-cu113

pip install opencv-python

cd /home/bpanigr/Research/fusion-network/scripts

## Execute your script
python pcl_main.py