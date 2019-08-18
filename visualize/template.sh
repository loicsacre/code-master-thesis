#!/bin/bash

python3 visualize/visualize-results-imagenet.py \
    --tissue lung-lobes_2 \
    --dye1 CD31 \
    --dye2 CC10 \
    --arch densenet201 \
    --size 300 \
    --references 3 7 \

