#!/bin/bash
#SBATCH --qos=low
#SBATCH --nodes=5
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=12G                             # Ask for 10 GB of RAM
#SBATCH --time=12:00:00                        # The job will run for 3 hours
#SBATCH --array=1-5
#SBATCH --mail-user=sharathraparthy@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -o /network/tmp1/raparths/slurm-%j.out

module load python/3.6
module load mujoco/2.0
module load cuda/10.0
source $HOME/sharath/bin/activate

mpirun -np 8 python -u train.py --seed=1 --sp-percent=0.5 --sp-polyak=0.95 --approach "adr"
