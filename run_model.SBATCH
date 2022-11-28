#!/bin/bash

#SBATCH --account=csci_ga_2572-2022fa
#SBATCH --partition=n1s16-v100-2
#SBATCH --time=1:00:00
#SBATCH --mem=25GB
#SBATCH --job-name=BACKBONE_SIMCLR_TRAINING
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_%j.out

singularity exec --nv --overlay /scratch/DL22FA/overlay_11-16.ext3:ro \
            -B /scratch/DL22FA/unlabeled_112.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python test.py"