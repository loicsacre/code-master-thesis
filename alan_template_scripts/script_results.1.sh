#!/bin/bash
#
#SBATCH --job-name=result
#SBATCH --output=/home/lsacre/outputs/result-%j.txt
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1


# if [[ "$1" == *"siamese"* ]]; then
#     echo "It's a Siamese network!"
#     srun python3 siamese/main.py -t --arch "$1" --checkpoint "$2" --output_dir results/final
# else
#     echo "It's a MatchNet network!"
#     srun python3 matchnet/main.py -t --arch "$1" --checkpoint "$2" --output_dir results/final
# fi

srun python3 evaluate.py --arch "$1" --checkpoint "$2" -d
