#!/bin/bash
#
#SBATCH --job-name=image1
#SBATCH --output=/home/lsacre/outputs/imagenet1-cos-seg-norm-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1


# srun python3 imagenet/1/main.py --arch alexnet --size 100
# srun python3 imagenet/1/main.py --arch resnet18 --size 100
# srun python3 imagenet/1/main.py --arch resnet34 --size 100
# srun python3 imagenet/1/main.py --arch resnet50 --size 100
# srun python3 imagenet/1/main.py --arch vgg16_bn --size 100
# srun python3 imagenet/1/main.py --arch vgg19_bn --size 100
# srun python3 imagenet/1/main.py --arch densenet161 --size 100
# srun python3 imagenet/1/main.py --arch densenet201 --size 100

srun python3 imagenet/1/main.py --arch alexnet --size 600
srun python3 imagenet/1/main.py --arch resnet18 --size 600
srun python3 imagenet/1/main.py --arch resnet34 --size 600
srun python3 imagenet/1/main.py --arch resnet50 --size 600
srun python3 imagenet/1/main.py --arch vgg16_bn --size 600
srun python3 imagenet/1/main.py --arch vgg19_bn --size 600
srun python3 imagenet/1/main.py --arch densenet161 --size 600
srun python3 imagenet/1/main.py --arch densenet201 --size 600

echo "With pooling for ResNet and DenseNet..."
srun python3 imagenet/1/main.py --arch resnet18 --size 600 -pool
srun python3 imagenet/1/main.py --arch resnet34 --size 600 -pool
srun python3 imagenet/1/main.py --arch resnet50 --size 600 -pool
srun python3 imagenet/1/main.py --arch densenet161 --size 600 -pool
srun python3 imagenet/1/main.py --arch densenet201 --size 600 -pool

# srun python3 imagenet/1/main.py --arch alexnet --size 100 --distance eucl
# srun python3 imagenet/1/main.py --arch resnet18 --size 100 --distance eucl
# srun python3 imagenet/1/main.py --arch resnet34 --size 100 --distance eucl
# srun python3 imagenet/1/main.py --arch resnet50 --size 100 --distance eucl
# srun python3 imagenet/1/main.py --arch vgg16_bn --size 100 --distance eucl
# srun python3 imagenet/1/main.py --arch vgg19_bn --size 100 --distance eucl
# srun python3 imagenet/1/main.py --arch densenet161 --size 100 --distance eucl
# srun python3 imagenet/1/main.py --arch densenet201 --size 100 --distance eucl

# srun python3 imagenet/1/main.py --arch alexnet --size 300 --distance eucl
# srun python3 imagenet/1/main.py --arch resnet18 --size 300 --distance eucl
# srun python3 imagenet/1/main.py --arch resnet34 --size 300 --distance eucl
# srun python3 imagenet/1/main.py --arch resnet50 --size 300 --distance eucl
# srun python3 imagenet/1/main.py --arch vgg16_bn --size 300 --distance eucl
# srun python3 imagenet/1/main.py --arch vgg19_bn --size 300 --distance eucl
# srun python3 imagenet/1/main.py --arch densenet161 --size 300 --distance eucl
# srun python3 imagenet/1/main.py --arch densenet201 --size 300 --distance eucl
