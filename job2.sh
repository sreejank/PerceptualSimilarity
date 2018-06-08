#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:55:00
#SBATCH --partition interactive
#SBATCH -o importing_output
module load GPU/Cuda/8.0
module load GPU/cuDNN
python test_import.py




