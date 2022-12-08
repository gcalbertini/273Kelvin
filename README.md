- ssh mr6744@greene.hpc.nyu.edu
- ssh burst
- srun --account=csci_ga_2572-2022fa -p interactive --pty /bin/bash
- cd /scratch_tmp/mr6744/273Kelvin/

- sbatch run_model.SBATCH
- squeue -u $USER
- cat slurm_100963.out
- scancel id

- :wq

- git reset HEAD -- SimCLR.pt

- scp model_2_batch_4_mom_0.9_decay_0.0001_epochs_11_lr_0.01_backbone_SimCLR_RPN.pt greene-dtn:/scratch/mr6744
- scp mr6744@greene.hpc.nyu.edu:/scratch/mr6744/model_2_batch_4_mom_0.9_decay_0.0001_epochs_11_lr_0.01_backbone_SimCLR_RPN.pt ~/Downloads