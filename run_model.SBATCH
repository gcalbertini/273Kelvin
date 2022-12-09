#!/bin/bash

#SBATCH --account=csci_ga_2572_2022fa_16
#SBATCH --partition=n1c10m64-v100-1
#SBATCH --time=00:45:00
#SBATCH --mem=60GB
#SBATCH --job-name=FASTRCNN
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

singularity exec --nv --overlay /scratch/DL22FA/overlay_12-02.ext3:ro \
            -B /scratch/DL22FA/unlabeled_224.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh;pip install lightning-bolts; python train.py --train_backbone -bbe 1 -bblr 0.003"
