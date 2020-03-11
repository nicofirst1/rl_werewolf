#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=rl_werewolf
#SBATCH --mem=800
module load Python/3.5.2-foss-2016a
module load tensorflow/1.5.0-foss-2016a-Python-3.5.2-CUDA-9.1.85
module load numpy/1.9.2-foss-2016a-Python-2.7.11

python test/parametric_action.py