#!/bin/bash
#
#SBATCH --job-name=matchnet
#SBATCH --output=/home/lsacre/outputs/matchnet-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1


# srun python3 master-thesis/matchnet/main.py --learning_rate "$1" --momentum "$2" --batch_size "$3" -d -v
srun python3 matchnet/main.py --learning_rate "$1" --momentum "$2" --batch_size "$3" --job_id "$SLURM_JOB_ID"