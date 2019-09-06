#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --gres gpu:1
#SBATCH -LXserver
#SBATCH --mincpus 52
#SBATCH --mem=42000
#SBATCH --reservation triesch-shared

##SBATCH --exclude springtalk

srun python3 asynchronous.py "$@"
