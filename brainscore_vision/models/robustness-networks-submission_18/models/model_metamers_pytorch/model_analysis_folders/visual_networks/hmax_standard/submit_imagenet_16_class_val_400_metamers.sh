#!/bin/bash
#SBATCH --job-name=met
#SBATCH --output=output/standard%A_%a.out
#SBATCH --error=output/standard%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-399
#SBATCH --dependency=after:19718639
#SBATCH --constraint=high-capacity
#SBATCH --exclude=node097,node094,node098,node093,node037
#SBATCH --partition=normal
module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
cp ../../../analysis_scripts/make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py .

# HMAX has different preprocessing and uses a loss with dropout
python make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py $SLURM_ARRAY_TASK_ID -I 3000 -L 'random_single_unit_optimization_inversion_loss_layer' -N 8 -DP -E 10 -Z 255

