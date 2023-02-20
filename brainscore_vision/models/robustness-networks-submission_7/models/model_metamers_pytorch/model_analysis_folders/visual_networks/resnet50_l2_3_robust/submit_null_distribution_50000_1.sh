#!/bin/bash
#SBATCH --job-name=null_resl2
#SBATCH --output=output/null_%A_%a.out
#SBATCH --error=output/null_%A_%a.err
#SBATCH --mem=128000
#SBATCH --cpus-per-task=20
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --array=0
#SBATCH --constraint=high-capacity
#SBATCH --partition=normal

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp ../../../analysis_scripts/make_null_distributions.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
# python -m pdb make_null_distributions.py -N 1000
# Run 5 random seeds, imagenet is randomized so we just have to choose a different RS each time
python make_null_distributions.py -N 50000 -R 0 & 
python make_null_distributions.py -N 50000 -R 1 &
python make_null_distributions.py -N 50000 -R 2 &
python make_null_distributions.py -N 50000 -R 3 &
python make_null_distributions.py -N 50000 -R 4 &

wait
