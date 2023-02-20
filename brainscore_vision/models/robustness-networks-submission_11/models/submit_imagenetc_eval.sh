#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%j.out
#SBATCH --error=output/behavior_plot%j.err
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-20%10
#SBATCH --constraint=11GB
#SBATCH --exclude=node093,node094,node097,node098
#SBATCH --partition=mcdermott

# List of the networks that we want to compare
NETWORK_LIST=("alexnet" "cornet_s" "resnet50" "resnet101" "vgg_19" "alexnet_l2_3_robust" "alexnet_linf_4_robust" "alexnet_linf_8_robust" "resnet50_byol" "resnet50_simclr" "resnet50_moco_v2" "resnet50_l2_3_robust" "resnet50_linf_4_robust" "resnet50_linf_8_robust" "alexnet_random_l2_3_perturb" "alexnet_random_linf8_perturb" "resnet50_random_l2_perturb" "resnet50_random_linf8_perturb" "alexnet_early_checkpoint" "alexnet_reduced_aliasing_early_checkpoint" "vonealexnet_gaussian_noise_std4_fixed" "vonelowpassalexnet_gaussian_noise_std4_fixed")

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3
source activate /om2/user/jfeather/conda_envs_files/brainscore_environment

BUILD_MODEL=${NETWORK_LIST[$SLURM_ARRAY_TASK_ID]}

python run_imagenet_c.py -n $BUILD_MODEL
