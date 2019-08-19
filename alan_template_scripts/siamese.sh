#!/bin/bash

# Train Siamese AlexNet

momemtum='0.0'
learning_rate='0.005'
batch_size='64'
margin='0.5'

sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"

# Test in another learning rate parameter
learning_rate='0.001'
sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"






