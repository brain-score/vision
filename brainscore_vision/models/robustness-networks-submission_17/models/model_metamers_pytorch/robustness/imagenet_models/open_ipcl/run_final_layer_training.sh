#!/bin/bash
#SBATCH --job-name=null_alexnetRA
#SBATCH --output=output/null_%A_%a.out
#SBATCH --error=output/null_%A_%a.err
#SBATCH --mem=64000
#SBATCH --cpus-per-task=20
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --array=0
#SBATCH --constraint=high-capacity
#SBATCH --partition=mcdermott

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3
source activate /om/user/jfeather/.conda/envs/model_metamers_pytorch_update_pytorch

python -m pdb main_lincls_onecycle.py ipcl1 fc7 --data /om2/data/public/imagenet/images_complete/ilsvrc/ --gpu 0

