#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=01:05:00
#SBATCH --partition day
#SBATCH -o gan_output
#SBATCH --mem-per-cpu 20G
module load GPU/Cuda/8.0
module load GPU/cuDNN
python gan.py




