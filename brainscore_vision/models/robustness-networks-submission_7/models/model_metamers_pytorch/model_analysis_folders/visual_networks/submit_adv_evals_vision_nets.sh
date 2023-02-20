#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%j.out
#SBATCH --error=output/behavior_plot%j.err
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=21
#SBATCH --constraint=11GB
#SBATCH --exclude=node093,node094,node097,node098
#SBATCH --partition=mcdermott

# List of the networks that we want to compare
NETWORK_LIST=("alexnet" "cornet_s" "resnet50" "resnet101" "vgg_19" "alexnet_l2_3_robust" "alexnet_linf_4_robust" "alexnet_linf_8_robust" "resnet50_byol" "resnet50_simclr" "resnet50_moco_v2" "resnet50_l2_3_robust" "resnet50_linf_4_robust" "resnet50_linf_8_robust" "alexnet_random_l2_3_perturb" "alexnet_random_linf8_perturb" "resnet50_random_l2_perturb" "resnet50_random_linf8_perturb" "alexnet_early_checkpoint" "alexnet_reduced_aliasing_early_checkpoint" "vonealexnet_gaussian_noise_std4_fixed" "vonelowpassalexnet_gaussian_noise_std4_fixed")

BUILD_MODEL=${NETWORK_LIST[$SLURM_ARRAY_TASK_ID]}
echo $BUILD_MODEL

# CD into the build model directory
cd $BUILD_MODEL

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp /om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/analysis_scripts_metamers_paper/ensemble_eval_range_eps_imagenet.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch

for RAND_SEED in 0 1 2 3 4; do 
    echo $RAND_SEED
    python ensemble_eval_range_eps_imagenet.py -R $RAND_SEED -N 1024 -I 64 -E 1 -B 32 -T '2' -D 4 -L -3 -M 1 -U 2
    python ensemble_eval_range_eps_imagenet.py -R $RAND_SEED -N 1024 -I 64 -E 1 -B 32 -T 'inf' -D 4 -U 2
    python ensemble_eval_range_eps_imagenet.py -R $RAND_SEED -N 1024 -I 64 -E 1 -B 32 -T '1' -D 4 -L 0 -M 3 -U 2
done
