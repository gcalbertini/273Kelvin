#!/bin/bash

#SBATCH --account=csci_ga_2572-2022fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=1:00:00
#SBATCH --mem=25GB
#SBATCH --job-name=BACKBONE
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

singularity exec --nv --overlay /scratch/DL22FA/overlay_11-16.ext3:ro \
            -B /scratch/DL22FA/unlabeled_224.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python lightning.py"


To make life easy, please just use —overlay flag for the 2 data squash files, the folders inside are  /labeled and  /unlabeled respectively,

singularity exec --nv --overlay /scratch/DL22FA/overlay_11-16.ext3:ro \
                      --overlay /scratch/DL22FA/unlabeled_224.sqsh:ro \
                      --overlay /scratch/DL22FA/labeled.sqsh:ro \
                      -B /scratch -B /scratch_tmp \
                      /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
                      /bin/bash


Singularity> find /labeled | wc -l
100007
Singularity> find /unlabeled | wc -l
512001
Singularity> 