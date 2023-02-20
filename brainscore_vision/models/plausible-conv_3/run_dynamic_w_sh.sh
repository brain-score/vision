#!/bin/bash

for trial in `seq 10 49`
do
  for kernel in 3 6 9
  do
    for gamma in 0.1 0.01 0.001
    do
#      COMMAND="python3 dynamic_weight_sharing_plots.py --kernel_size ${kernel} --gamma ${gamma} --trial ${trial}"
      sbatch --export=ALL,COMMAND="python3 dynamic_weight_sharing_plots.py --kernel-size ${kernel} --gamma ${gamma} --trial ${trial}" run_d_w_sh_slurm.sh
    done
  done
done