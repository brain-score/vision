#!/bin/bash
nohup python3 grid_search.py --gpu_name rtx5000 --experiment full_resnet18_imagenet --amp > logs_full_r18/conv &
nohup python3 grid_search.py --gpu_name a100 --experiment full_resnet18_imagenet --is-locally-connected --share-weights --instant-weight-sharing --weight-sharing-frequency 1 --amp > logs_full_r18/lc_wsh_1 &
nohup python3 grid_search.py --gpu_name a100 --experiment full_resnet18_imagenet --is-locally-connected --share-weights --instant-weight-sharing --weight-sharing-frequency 10 --amp > logs_full_r18/lc_wsh_10 &
nohup python3 grid_search.py --gpu_name a100 --experiment full_resnet18_imagenet --is-locally-connected --share-weights --instant-weight-sharing --weight-sharing-frequency 100 --amp > logs_full_r18/lc_wsh_100 &
nohup python3 grid_search.py --gpu_name a100 --experiment full_resnet18_imagenet --is-locally-connected --share-weights --instant-weight-sharing --weight-sharing-frequency 20 --amp > logs_full_r18/lc_wsh_20 &
nohup python3 grid_search.py --gpu_name a100 --experiment full_resnet18_imagenet --is-locally-connected --share-weights --instant-weight-sharing --weight-sharing-frequency 200 --amp > logs_full_r18/lc_wsh_200 &
nohup python3 grid_search.py --gpu_name a100 --experiment full_resnet18_imagenet --is-locally-connected --amp > logs_full_r18/lc &