#!/bin/bash
#
#SBATCH --job-name=image2d
#SBATCH --output=/home/lsacre/outputs/imagenet2_densenet201_"$1"-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1

srun python3 imagenet/2/main.py --size "$1" --shift "$2" --arch "$3"


