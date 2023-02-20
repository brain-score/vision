#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%A_%a.out
#SBATCH --error=output/behavior_plot%A_%a.out
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:QUADRORTX6000:1
#SBATCH --array=0-30
#SBATCH --constraint=11GB
#SBATCH --exclude=node112,node107
#SBATCH --partition=normal

# List of the networks that we want to compare
NETWORK_LIST=("texture_shape_alexnet_trained_on_SIN" "texture_shape_resnet50_trained_on_SIN" "CLIP_resnet50_float32" "CLIP_ViT-B_32_float32" "SWSL_resnet50" "SWSL_resnext101_32x8d" "vision_transformer_vit_large_patch16_224" "alexnet" "cornet_s" "resnet50" "resnet101" "vgg_19" "alexnet_l2_3_robust" "alexnet_linf_4_robust" "alexnet_linf_8_robust" "resnet50_byol" "resnet50_simclr" "resnet50_moco_v2" "resnet50_l2_3_robust" "resnet50_linf_4_robust" "resnet50_linf_8_robust" "alexnet_random_l2_3_perturb" "alexnet_random_linf8_perturb" "resnet50_random_l2_perturb" "resnet50_random_linf8_perturb" "alexnet_early_checkpoint" "alexnet_reduced_aliasing_early_checkpoint" "vonealexnet_gaussian_noise_std4_fixed")

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3
source activate /om2/user/jfeather/conda_envs_files/brainscore_environment

BUILD_MODEL=${NETWORK_LIST[$SLURM_ARRAY_TASK_ID]}

python run_imagenet_c.py -n $BUILD_MODEL
