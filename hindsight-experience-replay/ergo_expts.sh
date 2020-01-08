#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
                                              # SBATCH --nodes=5
                                              # SBATCH --ntasks=8
#SBATCH --mem=36G                             # Ask for 10 GB of RAM
#SBATCH --time=72:00:00                        # The job will run for 3 hours
#SBATCH --array=1-5
#SBATCH --mail-user=sharathraparthy@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH -o /network/home/raparths/onlysp-reacher-adr-ergo-%j.out

module load python/3.6
# module load mujoco/2.0
module load cuda/10.0
source $HOME/sharath/bin/activate

python  ddpg_train.py  --sp-percent 1.0 --approach 'unsupervised-default' --only-sp=True --env_name='ErgoReacherRandomizedEnv-Headless-v0' --n-params=8 
