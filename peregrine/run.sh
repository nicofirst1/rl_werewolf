#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=rl_ww
#SBATCH --mem=16GB
echo "Job started with path=$(pwd)"
module purge
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
bash scripts/train.sh src/trainable/evaluation_train.py
