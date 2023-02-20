#!/bin/bash
#SBATCH --job-name=null_hmax
#SBATCH --output=output/null_%A_%a.out
#SBATCH --error=output/null_%A_%a.err
#SBATCH --mem=64000
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0,10,20,30,40,50,60,70,80,90
#SBATCH --exclude=node093,node040,node097,node098,node094,node037
#SBATCH --constraint=high-capacity&11GB
#SBATCH --partition=normal

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp ../../../analysis_scripts/make_null_distributions.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch

python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+0)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+1)) --shuffle & 
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+2)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+3)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+4)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+5)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+6)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+7)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+8)) --shuffle &
python make_null_distributions.py -N 10000 -R $(($SLURM_ARRAY_TASK_ID+9)) --shuffle &

wait
