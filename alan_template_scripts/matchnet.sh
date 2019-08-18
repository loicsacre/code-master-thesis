#!/bin/bash

learning_rate='0.005'
batch_size='64'
momemtum='0.9'

sbatch scripts/script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"

batch_size='128'
learning_rate='0.005'
momemtum='0.9'
sbatch scripts/script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"

batch_size='128'
learning_rate='0.01'
momemtum='0.9'
sbatch scripts/script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"


# batch_size='64'
# sbatch script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"

# batch_size='128'
# sbatch script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"


# learning_rate='0.0001'
# sbatch script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"


# batch_size='128'

# momemtum='0.0'
# sbatch script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"

# momemtum='0.9'
# sbatch script_matchnet.sh "$learning_rate" "$momemtum" "$batch_size"



