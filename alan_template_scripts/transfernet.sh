#!/bin/bash

arch='transferVggnet'
learning_rate='0.01'
batch_size='64'
momemtum='0.9'

sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

learning_rate='0.001'
batch_size='64'
momemtum='0.9'

sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

learning_rate='0.001'
batch_size='64'
momemtum='0.0'

sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

learning_rate='0.01'
batch_size='128'
momemtum='0.0'

sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

learning_rate='0.01'
batch_size='128'
momemtum='0.9'

sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"


# arch='transferAlexnet'
# learning_rate='0.01'
# batch_size='64'
# momemtum='0.9'

# sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

# arch='transferAlexnet'
# learning_rate='0.001'
# batch_size='64'
# momemtum='0.9'

# sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

# arch='transferAlexnet'
# learning_rate='0.001'
# batch_size='64'
# momemtum='0.0'

# sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

# arch='transferAlexnet'
# learning_rate='0.01'
# batch_size='128'
# momemtum='0.0'

# sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"

# arch='transferAlexnet'
# learning_rate='0.01'
# batch_size='128'
# momemtum='0.9'

# sbatch scripts/script_transfernet.sh "$learning_rate" "$momemtum" "$batch_size" "$arch"


