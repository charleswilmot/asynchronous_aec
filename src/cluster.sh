4 #!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --gres gpu:3
#SBATCH -LXserver
#SBATCH --mincpus 46
#SBATCH --mem=90000
##SBATCH --reservation triesch-shared

##SBATCH --exclude springtalk

srun python3 asynchronous.py "$@"
