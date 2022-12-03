#!/bin/bash

### Request 1 node with 4 gpus, totals 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node (or ntasks in our 1-node case) and keep nodes 1 because we are billed hours proportional to Nodes*GPUs/Node*Time Job
### no multi-threading but for the deep learning codes TensorFlow and PyTorch, optimal performance can only be achieved when multiple CPU-cores are used to keep the GPU 
### busy by feeding it data. 125GB mem max on this node. 

#SBATCH --account=csci_ga_2572_2022fa_16
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # https://researchcomputing.princeton.edu/support/knowledge-base/pytorch on MNIST dataset dataloader with 4 workers and 4 here gave best performance
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=FULLRUN
#SBATCH --chdir=/scratch_tmp/ga_947/273Kelvin/
#SBATCH --output=slurm_%j.out


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

singularity exec --nv --overlay /scratch/DL22FA/overlay_12-02.ext3:ro \
            -B /scratch/DL22FA/unlabeled_224.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python train.py"
