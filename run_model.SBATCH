#!/bin/bash

### Request 1 node with 4 gpus, totals 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node (or ntasks in our 1-node case) and keep nodes 1 because we are billed hours proportional to Nodes*GPUs/Node*Wall_time
### no multi-threading but for the deep learning codes TensorFlow and PyTorch, optimal performance can only be achieved when multiple CPU-cores are used to keep the GPU 
### busy by feeding it dataloader. 125GB mem max on this node with 24 cores. 4 cpus/task * 1 task/node * 1 node / 24 CPU cores * 125 ~ 20GB node mem per GPU for total of 80GB estimate

#SBATCH --account=csci_ga_2572_2022fa_16
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --nodes=1                        # please keep this default
#SBATCH --ntasks=1                       # please keep this default
#SBATCH --ntasks-per-node=1              # please keep this default
#SBATCH --cpus-per-task=4    # https://researchcomputing.princeton.edu/support/knowledge-base/pytorch on MNIST dataset dataloader with 4 workers and 4 here gave best performance; assumed here too as little time to analyze optimal wall times
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=90GB
#SBATCH --job-name=FULLRUN
#SBATCH --output=slurm_%j.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ga947@nyu.edu


### Can change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
### https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
###export MASTER_PORT = 12340 but we are on GCP nodes so these are b-#-# as well, same as master address?
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


singularity exec --nv --overlay /scratch/DL22FA/overlay_12-02.ext3:ro \
            -B /scratch/DL22FA/unlabeled_224.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python train.py"
