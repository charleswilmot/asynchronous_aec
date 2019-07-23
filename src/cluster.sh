#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --gres gpu:1
#SBATCH -LXserver
#SBATCH --mincpus 16
#SBATCH --mem=30000
#SBATCH --exclude springtalk
#SBATCH --reservation triesch-shared

srun python3 asynchronous.py "$@"
