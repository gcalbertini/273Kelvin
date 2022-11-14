#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=load_evaluate
#SBATCH --gres=gpu:1

module purge

singularity exec --nv \
            --overlay /scratch/DL22FA/overlay-11-07.ext3:ro \
            -B /scratch/DL22FA/unlabeled-112.sqsh:/data:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/data:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python torch-test.py"
