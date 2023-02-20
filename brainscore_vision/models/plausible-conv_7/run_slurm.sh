#!/bin/bash
#SBATCH --time=30-00:00  # 30 days
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G

#module avail
#printenv
# submit from hpc-gw1
source /usr/share/modules/init/bash
module load cuda/11.2  # 10.1
source ~/.bashrc  # for pyenv
nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

#source ~/Studies/python_environments/gpu_venv/bin/activate
pyenv activate pyenv-3.7.3-pytorch1.9  # pyenv-3.7.3
cd ~/Studies/UCL/research_code/plausible-conv

echo $COMMAND
eval $COMMAND

pyenv deactivate
echo "Job finished."
exit
