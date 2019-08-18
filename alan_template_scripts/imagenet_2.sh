#!/bin/bash

size='600'
shift='150'

sbatch scripts/script_imagenet2.sh "$size" "$shift"
# sbatch scripts/script_imagenet2-alexnet.sh
# sbatch scripts/script_imagenet2-vgg19.sh
