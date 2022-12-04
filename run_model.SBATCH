#!/bin/bash

### Request 1 node with 4 gpus (to get WORLD_SIZE==4). No other configs allowed for this project though could've used 4 nodes with 1 GPU each so that ntasks/node is 1
### Note: --gres=gpu:x should equal to ntasks-per-node (or ntasks in our 1-node case) and keep nodes 1 because we are billed hours proportional to Nodes*GPUs/Node*Wall_time
### no multi-threading but for the deep learning codes TensorFlow and PyTorch, optimal performance can only be achieved when multiple CPU-cores are used to keep the GPU 
### busy by feeding it dataloader. 125GB mem max on this node with 24 cores. 4 cpus/task * 1 task/node * 1 node / 24 CPU cores * 125 ~ 20GB node mem per GPU for total of 80GB estimate

#SBATCH --account=csci_ga_2572_2022fa_16 
#SBATCH --cpus-per-task=4               # https://researchcomputing.princeton.edu/support/knowledge-base/pytorch on MNIST dataset dataloader with 4 workers and 4 here gave best performance; assumed here too as little time to analyze optimal wall times
#SBATCH --partition=n1c24m128-v100-4    # Proper partition for num GPUs/node: n1s8-v100-1, n1s16-v100-2, or n1c24m128-v100-4
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks-per-node=4             # total number of tasks per node (or --ntasks=4 for the single node we are allowed)          
#SBATCH --gres=gpu:4                    # 4 GPUs per node
#SBATCH --time=0:40:00
#SBATCH --mem=90GB
#SBATCH --job-name=FULLRUN
#SBATCH --output=slurm_%j.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ga947@nyu.edu


### Alternatively, you could request --nodes=4 with 1 GPU each by setting nodes to 4 and --ntasks-per-node=1 OR --nodes=2 with 2 GPUs each. Just use proper partition and GPUs.
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# srun included as each process generated for us, else have to include mp.spawn in code using main
srun singularity exec --nv --overlay /scratch/DL22FA/overlay_12-02.ext3:ro \
            -B /scratch/DL22FA/unlabeled_224.sqsh:/unlabeled:image-src=/ \
            -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
            -B /scratch -B /scratch_tmp \
            /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python train.py"
