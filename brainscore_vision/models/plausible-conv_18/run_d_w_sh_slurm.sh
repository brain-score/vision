#!/bin/bash
#SBATCH --time=1-00:00  # 30 days
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --job-name=dwsh
#SBATCH --output="/ceph/scratch/romanp/plausible-conv/logs_d_w_sh/logs.out"

source ~/.bashrc
pyenv activate pyenv-3.7.3
cd ~/Studies/UCL/research_code/plausible-conv
eval $COMMAND
