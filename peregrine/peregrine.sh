#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=rl_werewolf
#SBATCH --mem=800
module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
python test/parametric_action.py