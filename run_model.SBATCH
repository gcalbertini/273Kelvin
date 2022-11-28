#!/bin/bash

#SBATCH --account=csci_ga_2572-2022fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=BACKBONE
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

singularity exec --nv --overlay /scratch/DL22FA/overlay_11-16.ext3:ro \
            -B /scratch/DL22FA/unlabeled_112.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python train_simclr.py"