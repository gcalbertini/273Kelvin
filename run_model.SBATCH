#!/bin/bash

#SBATCH --account=csci_ga_2572_2022fa_16
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=FULLRUN
#SBATCH --gres=gpu:4
#SBATCH --output=slurm_%j.out

# --- Setup In Master ---
# Master will be the one running this bash script (SLURM runs this only once)
# Get hostname and port on first node first process
# For the port see: https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
master_addr=$(hostname -i)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

singularity exec --nv --overlay /scratch/DL22FA/overlay_11-16.ext3:ro \
            -B /scratch/DL22FA/unlabeled_224.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python train.py --train_backbone --master_addr $master_addr --master_port $master_port"
