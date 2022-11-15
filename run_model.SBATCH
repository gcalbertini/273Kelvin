#!/bin/bash

#SBATCH --account=csci_ga_2572-2022fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=load_test
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

module purge

singularity exec --nv --overlay /scratch/DL22FA/overlay_11-15.ext3:ro \
            -B /scratch/DL22FA/unlabeled_112.sqsh:/labeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/unlabeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -i "source /ext3/env.sh; python /scratch/$USER/273Kelvin/torch-test.py"
