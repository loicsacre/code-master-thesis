#!/bin/bash
#
#SBATCH --job-name=image2r
#SBATCH --output=/home/lsacre/output/output_imagenet2_resnet.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

# srun python3 master-thesis/slice/main.py -pool --size 300
# srun python3 master-thesis/slice/main.py -pool --size 300 --arch alexnet
# srun python3 master-thesis/slice/main.py --size 300 --arch densenet201

# srun python3 master-thesis/slice/main.py --size 100 --arch alexnet
# srun python3 master-thesis/slice/main.py --size 300 --arch densenet201
# srun python3 master-thesis/slice/main.py --size 300 --arch resnet50
# srun python3 master-thesis/imagenet/2/main-distance.py --size 300 --arch resnet50
# srun python3 master-thesis/imagenet/2/main-distance.py --size 300 --arch densenet201

srun python3 master-thesis/imagenet/2/main-distance.py --size 300 --arch resnet50


