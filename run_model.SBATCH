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
            --overlay /scratch/<$USER>/capstone/overlay-50G-10M.ext3:ro \
            --overlay /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04-20211015.sqf:ro \
            /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python <path_to_python_script.py>"