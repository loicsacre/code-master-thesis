#!/bin/bash
#
#SBATCH --job-name=transfernet
#SBATCH --output=/home/lsacre/outputs/transfernet-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# srun python3 master-thesis/matchnet/main.py --arch transfernet --learning_rate "$1" --momentum "$2" --batch_size "$3"
# srun python3 master-thesis/matchnet/main.py -o --learning_rate 0.006
# srun python3 master-thesis/matchnet/main.py -o --learning_rate 0.003

# srun python3 master-thesis/matchnet/main.py --arch transferAlexnet --learning_rate "$1" --momentum "$2" --batch_size "$3"  --job_id "$SLURM_JOB_ID"
srun python3 matchnet/main.py --learning_rate "$1" --momentum "$2" --batch_size "$3" --arch "$4" --job_id "$SLURM_JOB_ID"