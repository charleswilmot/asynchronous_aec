#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --gres gpu:1
#SBATCH -LXserver
#SBATCH --mincpus 44
#SBATCH --mem=100000
##SBATCH --reservation triesch-shared

##SBATCH --exclude springtalk

srun python3 asynchronous.py "$@"
