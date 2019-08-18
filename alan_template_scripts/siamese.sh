#!/bin/bash

learning_rate='0.005'
batch_size='64'
momemtum='0.9'
margin='0.2'

echo "$margin"
sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"

margin='0.5'
echo "$margin"
sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"

margin='1.0'
echo "$margin"
sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"

margin='1.5'
echo "$margin"
sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"



# margin='0.5'
# sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"

# margin='2.0'
# sbatch scripts/script_siamese.sh "$learning_rate" "$momemtum" "$batch_size" "$margin"






