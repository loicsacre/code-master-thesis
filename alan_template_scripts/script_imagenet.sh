#!/bin/bash
#
#SBATCH --job-name=image1
#SBATCH --output=/home/lsacre/outputs/imagenet1-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1

srun python3 imagenet/1/main.py --arch alexnet --size 300
srun python3 imagenet/1/main.py --arch resnet18 --size 300
srun python3 imagenet/1/main.py --arch resnet34 --size 300
srun python3 imagenet/1/main.py --arch resnet50 --size 300
srun python3 imagenet/1/main.py --arch vgg16_bn --size 300
srun python3 imagenet/1/main.py --arch vgg19_bn --size 300
srun python3 imagenet/1/main.py --arch densenet161 --size 300
srun python3 imagenet/1/main.py --arch densenet201 --size 300

echo "With pooling for ResNet and DenseNet..."

srun python3 imagenet/1/main.py --arch resnet18 --size 300 -pool
srun python3 imagenet/1/main.py --arch resnet34 --size 300 -pool
srun python3 imagenet/1/main.py --arch resnet50 --size 300 -pool
srun python3 imagenet/1/main.py --arch densenet161 --size 300 -pool
srun python3 imagenet/1/main.py --arch densenet201 --size 600 -pool
