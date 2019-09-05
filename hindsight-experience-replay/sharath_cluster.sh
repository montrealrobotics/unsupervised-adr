#!/bin/bash
#SBATCH --qos=high                      # Ask for unkillable job
#SBATCH --cpus-per-task=8                     # Ask for 2 CPUs
#SBATCH --gres=gpu:0                          # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --time=6:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/raparths/slurm-%j.out  # Write the log on tmp1

# 1. Load your environment
conda activate poppy

# 2. Copy your dataset on the compute node
# cp /network/data/<dataset> $SLURM_TMPDIR
echo $SLURM_TMPDIR

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
mpirun -np 8 python -u train.py --sp-percent=0.0 --sp-polyak=0.95 --seed 128  --approach "udr" 2>&1 --path $SLURM_TMPDIR | tee reach.log

# 4. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/saved_models /network/tmp1/raparths/