#!/bin/bash

# Launch the second experiment for approach 1 with pre-trained network, size and shift 
# parameters for the segmentation

size='300'
shift='75'
arch='densenet201'

sbatch scripts/script_imagenet2.sh "$size" "$shift" "$densenet201"

