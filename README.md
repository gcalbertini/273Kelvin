- ssh mr6744@greene.hpc.nyu.edu
- ssh burst
- srun --account=csci_ga_2572-2022fa -p interactive --pty /bin/bash
- cd /scratch_tmp/mr6744/273Kelvin/

- sbatch run_model.SBATCH 
- squeue -u $USER
- cat slurm_100963.out
- scancel id

- :wq
- ls

- git reset HEAD -- SimCLR.pt