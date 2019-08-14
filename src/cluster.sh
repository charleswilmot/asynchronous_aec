#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --gres gpu:1
#SBATCH -LXserver
#SBATCH --mincpus 52
#SBATCH --mem=50000
#SBATCH --exclude springtalk
#SBATCH --reservation triesch-shared

srun python3 asynchronous.py "$@"
