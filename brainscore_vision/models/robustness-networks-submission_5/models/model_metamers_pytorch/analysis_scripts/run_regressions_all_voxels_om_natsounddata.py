"""
Voxel prediction analysis using fMRI data from Norman-Haignere et al. 2015
Follows same analysis as used in Kell et al. 2018

Please cite the assciated papers if you use this analysis pipeline. 

@article{norman2015distinct,
  title={Distinct cortical pathways for music and speech revealed by hypothesis-free voxel decomposition},
  author={Norman-Haignere, Sam and Kanwisher, Nancy G and McDermott, Josh H},
  journal={Neuron},
  volume={88},
  number={6},
  pages={1281--1296},
  year={2015},
  publisher={Elsevier}
}

@article{kell2018task,
  title={A task-optimized neural network replicates human auditory behavior, predicts brain responses, and reveals a cortical processing hierarchy},
  author={Kell, Alexander JE and Yamins, Daniel LK and Shook, Erica N and Norman-Haignere, Sam V and McDermott, Josh H},
  journal={Neuron},
  volume={98},
  number={3},
  pages={630--644},
  year={2018},
  publisher={Elsevier}
}
"""

import h5py
import numpy as np
import pickle

import sklearn
from sklearn import *
import scipy

import os
import sys

from voxel_regression_functions import * 
from analysis_scripts.default_paths import fMRI_DATA_PATH

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
###### INPUT ARGUMENTS ######
#########PARSE THE ARGUMENTS FOR THE FUNCTION#########
parser = argparse.ArgumentParser(description='Input the index to choose the layer')
parser.add_argument('LAYER',metavar='L',type=int,help='index into the list of possible layers')
parser.add_argument('FEATURES',metavar='F',type=str, help='path to the feature file')
parser.add_argument('NUMSPLITS',metavar='--S',type=int,nargs='?',default=10,help='Number of splits to run on the dataset')
parser.add_argument('RANDSEED',metavar='--D',type=int,nargs='?',default=3882,help='Random seed to use for determining the splits')
parser.add_argument('OVERWRITE', metavar='--O', type=str2bool, nargs='?', const=True, default=False, help='Force overwrite pickle files')
parser.add_argument('-Z', '--ZERO_CENTER', action='store_true', help='If true, zero mean the targets')

args=parser.parse_args()

layer_idx = args.LAYER
num_random_splits = args.NUMSPLITS
rand_seed = args.RANDSEED
feature_file = args.FEATURES
overwrite_pckl = args.OVERWRITE
zero_center = args.ZERO_CENTER

np.random.seed(rand_seed)
features_time_avg = h5py.File(feature_file,'r')
###### END INPUT ARGUMENTS ######

###### OTHER PARAMETERS  that are changed infrequently ######
n_for_train = 83
n_for_test = 82

possible_alphas = [10**x for x in range(-40,40)]
# possible_alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20, 100, 200, 1000, 2000]
possible_alphas = possible_alphas[::-1] # reverse the order to make it consistant with Kell2018's conventions

EXPORT_DIR = 'regression_results/'
if not os.path.isdir(EXPORT_DIR):
    os.mkdir(EXPORT_DIR)
###### END OTHER PARAMETERS ######

###### CHANGE THIS SECTION WHEN USING DIFFERENT MEASURED MODEL FEATURES ######
# load in the features
# here, we are loading in alex's networks.
layers = features_time_avg['layer_list']
layer = layers[layer_idx]
print(layer)
EXPORT_DIR_FULL = os.path.join(EXPORT_DIR, feature_file.split('/')[-1].split('.h5')[0])
if not os.path.isdir(EXPORT_DIR_FULL):
    os.mkdir(EXPORT_DIR_FULL)

if zero_center:
    save_path = os.path.join(EXPORT_DIR_FULL, 'voxelwise_regression_%s_randseed%d_zero_center_fmri_activations.pckl'%(layer, rand_seed))
else:
    save_path = os.path.join(EXPORT_DIR_FULL, 'voxelwise_regression_%s_randseed%d.pckl'%(layer, rand_seed))
###### END OF MODEL SPECIFIC PARAMETERS ######

### Check if the pickle already exists
if os.path.isfile(save_path) and not overwrite_pckl:
    raise FileExistsError('The file %s already exists, and you are not forcing overwriting')

# Where the sound names are stored (although we shouldn't need them, because everything is in the same order)
sound_meta = np.load(os.path.join(fMRI_DATA_PATH, 'neural_stim_meta.npy'))

# Where the voxel data lives, and only get the voxels with 3 runs. 
voxel_data_all = np.load(os.path.join(fMRI_DATA_PATH, 'voxel_features_array.npy'))
voxel_meta_all = np.load(os.path.join(fMRI_DATA_PATH, 'voxel_features_meta.npy'))
is_3 = voxel_meta_all['n_reps'] == 3
voxel_meta, voxel_data = voxel_meta_all[is_3], voxel_data_all[:,is_3,:]

# Voxel_meta contains the participant ID ('subj_idx') and the coordinates for plotting the voxels
# Run the analysis on all voxels that had 3 runs (7694 voxels) and then afterwards choose which ones we are plotting based on the ROI. 
# Voxel data is shape [165 stimuli, number voxels, number runs]

n_stim = voxel_data_all.shape[0]

# ROI masks, these are already excluding voxels from subjets with only 2 runs
if sys.version[0] == '2': 
    roi_masks = pickle.load(open(os.path.join(fMRI_DATA_PATH, 'roi_masks.cpy'),'rb'))
else: 
    roi_masks = pickle.load(open(os.path.join(fMRI_DATA_PATH, 'roi_masks.cpy','rb')), encoding='latin1')

r2_voxels = np.empty([voxel_data.shape[1], num_random_splits])
alphas = np.empty([voxel_data.shape[1], num_random_splits])
all_train_idxs = np.empty([n_stim, num_random_splits])
all_test_idxs = np.empty([n_stim, num_random_splits])

for split_idx in range(num_random_splits):
    print('SPLIT %d'%split_idx)

    train_data_idxs = np.random.choice(n_stim, size=n_for_train, replace=False)
    set_of_possible_test_idxs = set(np.arange(n_stim)) - set(train_data_idxs)
    test_data_idxs = np.random.choice(list(set_of_possible_test_idxs), size=n_for_test, replace=False)
    is_train_data, is_test_data = np.zeros((n_stim), dtype=bool), np.zeros((n_stim), dtype=bool)
    is_train_data[train_data_idxs], is_test_data[test_data_idxs] = True, True

    all_train_idxs[:,split_idx] = is_train_data.copy()
    all_test_idxs[:,split_idx] = is_test_data.copy()

    features = features_time_avg[layer]
        
    for voxel_idx in range(voxel_data.shape[1]):
        print(voxel_idx)
        r2_voxels[voxel_idx, split_idx], alphas[voxel_idx, split_idx] = runRidgeWithCorrectedR2_ThreeRunSplit(features, 
                                                      voxel_data, 
                                                      voxel_idx, 
                                                      is_train_data, 
                                                      is_test_data, 
                                                      possible_alphas,
                                                      zero_center=zero_center)
                
## things to save in the dictionary save the voxel idxs for the splits for reproducability
info = { 'r2s': r2_voxels,
         'alphas': alphas,
         'layer_name': layer,
         'random_seed': rand_seed,
         'all_train_idxs':all_train_idxs,
         'all_test_idxs':all_test_idxs,
         'feature_file':feature_file,
         'roi_masks':roi_masks,
         'voxel_meta':voxel_meta,
        }

pickle.dump(info, open(save_path, 'wb'))
