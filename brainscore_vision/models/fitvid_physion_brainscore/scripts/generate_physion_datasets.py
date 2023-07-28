import os
from  glob import glob
import numpy as np

pretraining_train_dir = '/mnt/fs4/eliwang/physion_data/pretraining_train'
dataset_sizes = [7, 13, 32, 63, 125, 625, 1250]
num_datasets = 3

# get files
scenario_dirs = glob(os.path.join(pretraining_train_dir, '*', ''))

# shuffle then split into datasets
for scenario_dir in scenario_dirs:
    scenario_name = scenario_dir.split('/')[-2]
    print(scenario_name)
    files = glob(os.path.join(scenario_dir, '*.hdf5'))
    for dataset_size in dataset_sizes:
        rng = np.random.RandomState(seed=dataset_size)
        files_shuffle = np.array(files, copy=True)
        rng.shuffle(files_shuffle)
        num_files = len(files_shuffle)
        num_splits = min(num_datasets, num_files//dataset_size)
        for i in range(num_splits):
            selected_files = files_shuffle[i*dataset_size:(i+1)*dataset_size]
            dst_dir = scenario_dir.replace(scenario_name, f'{scenario_name}_size{dataset_size}_split{i}')
            os.mkdir(dst_dir)
            for src in selected_files:
                dst = src.replace(scenario_name, f'{scenario_name}_size{dataset_size}_split{i}')
                # print(src, dst)
                os.symlink(src, dst)
