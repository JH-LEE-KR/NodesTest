#!/bin/bash

#SBATCH --job-name=admin_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH -p admin
#SBATCH -w agi1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=1-0
#SBATCH -o %N_%x_%j.out
#SBTACH -e %N_%x_%j.err

source
conda

python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py