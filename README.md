# 273Kelvin
NYU Deep Learning Project

Steps to have the model:
1. Build the SimCLR model and train on unlabelled data
2. Freeze the weight of the trained SimCLR
3. Transfer the weight to Bounding Box model and train it on labelled data
4. Save the trained bounding box model and submit it
0. Ensure your git SSH keys are set up in Burst
1. On NYU network or VPN, `ssh net_id@burst.hpc.nyu.edu`
2. `srun --account=csci_ga_2572-2022fa -p interactive --pty /bin/bash`
3. `git clone https://github.com/gcalbertini/273Kelvin.git` (ensure you got the latest)
4. Navigate to the branch you're working on
5. `sbatch run_model.SBATCH`
	1. Change script path as needed
    2. `squeue -u $USER` to see progress of job
6. Look for `slurm-#####.out` file by typing `ls` in running directory
