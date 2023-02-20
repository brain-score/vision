#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%j.out
#SBATCH --error=output/behavior_plot%j.err
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --time=01:20:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-22
#SBATCH --constraint=11GB
#SBATCH --exclude=node093,node094,node097,node098
#SBATCH --partition=normal

# List of the networks that we want to compare
NETWORK_LIST=("alexnet" "cornet_s" "resnet50" "resnet101" "vgg_19" "alexnet_l2_3_robust" "alexnet_linf_4_robust" "alexnet_linf_8_robust" "resnet50_byol" "resnet50_simclr" "resnet50_moco_v2" "resnet50_l2_3_robust" "resnet50_linf_4_robust" "resnet50_linf_8_robust" "alexnet_random_l2_3_perturb" "alexnet_random_linf8_perturb" "resnet50_random_l2_perturb" "resnet50_random_linf8_perturb" "alexnet_early_checkpoint" "alexnet_reduced_aliasing_early_checkpoint" "vonealexnet_gaussian_noise_std4_fixed")

BUILD_MODEL=${NETWORK_LIST[$SLURM_ARRAY_TASK_ID]}
echo $BUILD_MODEL

# CD into the build model directory
cd $BUILD_MODEL

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp ../../../analysis_scripts/imagenet_network_network_comparisons.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch

for MET_MODEL in ${NETWORK_LIST[@]}; do
    echo $MET_MODEL
    python imagenet_network_network_comparisons.py $MET_MODEL
done

