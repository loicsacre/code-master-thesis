#!/bin/bash

arch='matchnet'
checkpoint='./results/matchnet/matchnet/checkpoints/matchnet_0.005_0.9_64-1210032.check'

sbatch scripts/script_results.sh "$arch" "$checkpoint"

arch='transferVggnet'
checkpoint='./results/matchnet/transferVggnet/checkpoints/transferVggnet_0.001_0.9_64-1210036.check'

sbatch scripts/script_results.sh "$arch" "$checkpoint"

arch='transferAlexnet'
checkpoint='./results/matchnet/transferAlexnet/checkpoints/transferAlexnet_0.001_0.9_64-1210028.check'

sbatch scripts/script_results.sh "$arch" "$checkpoint"

arch='siameseAlexnet'
checkpoint='./results/siamese/siameseAlexnet/checkpoints/siameseAlexnet_0.01_0.9_64-1205599.check'

sbatch scripts/script_results.sh "$arch" "$checkpoint"
# sbatch scripts/script_results.1.sh "$arch" "$checkpoint"
