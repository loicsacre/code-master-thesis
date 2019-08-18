#!/bin/bash
#
#SBATCH --job-name=evaluate
#SBATCH --output=/home/lsacre/outputs/evaluate-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1


srun python3 evaluate.py --arch "$1" --checkpoint "$2"
# srun python3 evaluate-sum.py --arch "$1" --checkpoint "$2"