0. Ensure your git SSH keys are set up in Burst
1. On NYU network or VPN, `ssh net_id@burst.hpc.nyu.edu`
2. `srun --account=csci_ga_2572-2022fa -p interactive --pty /bin/bash`
3. `git clone https://github.com/gcalbertini/273Kelvin.git` (ensure you got the latest)
4. Navigate to the branch you're working on
5. `sbatch run_model.SBATCH`
	1. Change script path as needed
    2. `squeue -u $USER` to see progress of job
6. Look for `slurm-#####.out` file by typing `ls` in running directory