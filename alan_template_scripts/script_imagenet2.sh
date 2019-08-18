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

# srun python3 master-thesis/slice/main.py -pool --size 300
# srun python3 master-thesis/slice/main.py -pool --size 300 --arch alexnet
# srun python3 master-thesis/slice/main.py --size 300 --arch densenet201

# srun python3 master-thesis/slice/main.py -pool --size 100 --arch alexnet
# srun python3 master-thesis/slice/main.py --size 300 --arch densenet201
# srun python3 master-thesis/slice/main.py --size 300 --arch resnet50
# srun python3 master-thesis/imagenet/2/main-distance.py --size 300 --arch resnet50


# image2d
# /home/lsacre/output/output_imagenet2_densenet-%j.txt


srun python3 imagenet/2/main.py --size "$1" --arch densenet201 --shift "$2"

# srun python3 imagenet/2/main.py --size 200 --arch densenet201 --shift 50
# srun python3 imagenet/2/main.py --size 400 --arch densenet201 --shift 100
# srun python3 imagenet/2/main.py --size 500 --arch densenet201 --shift 125
# srun python3 imagenet/2/main.py --size 700 --arch densenet201 --shift 175
# srun python3 imagenet/2/main.py --size 800 --arch densenet201 --shift 200
# srun python3 imagenet/2/main.py --size 900 --arch densenet201 --shift 225
# srun python3 imagenet/2/main.py --size 1000 --arch densenet201 --shift 250
# srun python3 imagenet/2/main.py --size 1100 --arch densenet201 --shift 275
# srun python3 imagenet/2/main.py --size 1200 --arch densenet201 --shift 300

# srun python3 imagenet/2/main-test.py --size 600 --arch densenet201 --shift 150

# srun python3 imagenet/2/main-distance.py --size 300 --arch densenet201

# rotation
# /home/lsacre/output/output_imagenet2_rotation-%j.txt
# srun python3 master-thesis/experiments/imagenet-rotation.py --size 300 --arch densenet201



