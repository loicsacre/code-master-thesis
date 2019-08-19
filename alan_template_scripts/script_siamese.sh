#!/bin/bash
#
#SBATCH --job-name=siamese
#SBATCH --output=/home/lsacre/outputs/siamese-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

srun python3 siamese/main.py --learning_rate "$1" --momentum "$2" --batch_size "$3" --margin "$4" --job_id "$SLURM_JOB_ID"

