#!/bin/bash
#SBATCH --job-name=met_resnet50l23
#SBATCH --output=output/standard%A_%a.out
#SBATCH --error=output/standard%A_%a.err
#SBATCH --mem=4000
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --constraint=high-capacity
#SBATCH --exclude=node093,node040
#SBATCH --partition=mcdermott

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
cp /om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/analysis_scripts_metamers_paper/make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py .
python make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py 32 -I 3000 -N 8 -R $SLURM_ARRAY_TASK_ID
