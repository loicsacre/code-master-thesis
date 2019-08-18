#!/bin/bash
#
#SBATCH --job-name=rotation
#SBATCH --output=/home/lsacre/outputs/imagenet2_rotation-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# srun ./scripts/sleep.sh
# srun python3 datasets/normalization.py

# rotation
# /home/lsacre/output/output_imagenet2_rotation-%j.txt
srun python3 experiments/imagenet-rotation.py --size 300 --arch densenet201

