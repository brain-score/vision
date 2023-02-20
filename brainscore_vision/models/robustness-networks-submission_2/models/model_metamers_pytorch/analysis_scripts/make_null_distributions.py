"""
Creates a null distribution for a model created with a build_network.py file from robust audio. 

Works for both image and audio clips, using the dataset and augmentations that are specified in the build_network.py file. 
"""

import torch

from robustness.tools.distance_measures import *

from matplotlib import pylab as plt
import numpy as np

import importlib.util
import scipy

from scipy.io import wavfile

import time
import argparse
import pickle
import os
    
def _get_activations(all_outputs, metamer_layers):
    activations_1 = []
    activations_2 = []
    for layer in metamer_layers:
        if isinstance(all_outputs[layer], list):
            activations_1.append(np.concatenate([out_value[0,:].detach().cpu().numpy().ravel().astype(np.float32) for out_value in all_outputs[layer]], 0))
            activations_2.append(np.concatenate([out_value[1,:].detach().cpu().numpy().ravel().astype(np.float32) for out_value in all_outputs[layer]], 0))
        else:
            activations_1.append(all_outputs[layer][0,:].detach().cpu().numpy().ravel().astype(np.float32))
            activations_2.append(all_outputs[layer][1,:].detach().cpu().numpy().ravel().astype(np.float32))
    return activations_1, activations_2

def run_null_distribution(NUMNULL, SPLITIDX, PATHNULL, RANDOMSEED, OVERWRITE_PICKLE, shuffle, MODEL_DIRECTORY):
    BATCH_SIZE=2
    NUM_WORKERS=2
    STARTIDX = (NUMNULL*BATCH_SIZE)*SPLITIDX

    if MODEL_DIRECTORY is None:
        import build_network
        MODEL_DIRECTORY = '' # use an empty string to append to saved files.
    else:
        build_network_spec = importlib.util.spec_from_file_location("build_network",
            os.path.join(MODEL_DIRECTORY, 'build_network.py'))
        build_network = importlib.util.module_from_spec(build_network_spec)
        build_network_spec.loader.exec_module(build_network)

    pckl_path = os.path.join(MODEL_DIRECTORY, PATHNULL, 'null_dist_N_%d_RS_%d_SPLIT_%d_START_%d.pckl'%(NUMNULL, RANDOMSEED, SPLITIDX, STARTIDX))
    if os.path.isfile(pckl_path) and not OVERWRITE_PICKLE:
        raise FileExistsError('The file %s already exists, and you are not forcing overwriting'%pckl_path)
    
    print('Shuffle set to: ', shuffle)
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    print('SPLIT %d'%SPLITIDX)
    print('RANDOM SEED %d'%RANDOMSEED)
    
    if shuffle:
        subset_type = 'rand'
    else:
        subset_type = 'first'

    model, ds, metamer_layers = build_network.main(return_metamer_layers=True)

    train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE,
                                               workers=NUM_WORKERS,
                                               subset=NUMNULL*BATCH_SIZE,
                                               subset_type=subset_type,
                                               subset_start=STARTIDX,
                                               shuffle_train=shuffle, # Need to shuffle to get different null samples each time. 
                                               shuffle_val=shuffle,
                                               data_aug=True,
                                              )
    data_iterator = enumerate(train_loader)
    
    # Send model to GPU (b/c we haven't loaded a model, so it is not on the GPU)
    model = model.cuda()
    model.eval()
    
    spearman_r = []
    pearson_r = []
    dB_snr_values = []
    norm_noise = []
    norm_signal = []
    
    for null_iter in range(NUMNULL):
        _, (im, targ) = next(data_iterator) # Images to invert
        with torch.no_grad():
            (predictions, rep, all_outputs), orig_image = model(im.cuda(), with_latent=True, fake_relu=True) # Corresponding representation
        activations_1, activations_2 = _get_activations(all_outputs, metamer_layers)
        spearman_r_pair = []
        pearson_r_pair = []
        dB_snr_values_pair = []
        norm_noise_pair = []
        norm_signal_pair = []

        for layer_idx, layer in enumerate(metamer_layers):
            if (np.sum(activations_1[layer_idx].ravel())==0) or (np.sum(activations_2[layer_idx].ravel())==0):
                print('FOUND ALL ZEROS FOR LAYER %s, null_iter %d, SPLIT %d, SEED %d'%(layer, null_iter, 
                           SPLITIDX, RANDOMSEED))
                # If one of the images gives all zero activations (which can happen with some crops) 
                # then set the correlations to zero
                spearman_r_pair.append(0)
                pearson_r_pair.append(0)
                # db_SNR_i may be -inf if the signal is all zeros. So we set this value to 0 as well (signal=noise)
                db_SNR_i, norm_signal_i, norm_noise_i = compute_snr_db([activations_1[layer_idx], activations_2[layer_idx]])
                dB_snr_values_pair.append(0)
                norm_signal_pair.append(norm_signal_i)
                norm_noise_pair.append(norm_noise_i)
            else:
                spearman_r_pair.append(compute_spearman_rho_pair([activations_1[layer_idx], activations_2[layer_idx]]))
                pearson_r_pair.append(compute_pearson_r_pair([activations_1[layer_idx], activations_2[layer_idx]]))
                db_SNR_i, norm_signal_i, norm_noise_i = compute_snr_db([activations_1[layer_idx], activations_2[layer_idx]]) 
                dB_snr_values_pair.append(db_SNR_i)
                norm_signal_pair.append(norm_signal_i)
                norm_noise_pair.append(norm_noise_i)

        spearman_r.append(spearman_r_pair)
        pearson_r.append(pearson_r_pair)
        dB_snr_values.append(dB_snr_values_pair)
        norm_signal.append(norm_signal_pair)
        norm_noise.append(norm_noise_pair)
        if null_iter%10==0:
            print('{:f} percent done'.format(null_iter/NUMNULL), flush=True)
    
    spearman_r = np.array(spearman_r)
    pearson_r = np.array(pearson_r)
    dB_snr_values = np.array(dB_snr_values)
    norm_signal = np.array(norm_signal)
    norm_noise = np.array(norm_noise)
 
    spearman_r_dict = {}
    pearson_r_dict = {}
    dB_snr_values_dict = {}
    norm_signal_dict = {}
    norm_noise_dict = {}
    for layer_idx, layer in enumerate(metamer_layers):
        spearman_r_dict[layer] = spearman_r[:,layer_idx]
        pearson_r_dict[layer] = pearson_r[:,layer_idx]
        norm_signal_dict[layer] = norm_signal[:,layer_idx]
        norm_noise_dict[layer] = norm_noise[:,layer_idx]
        dB_snr_values_dict[layer] = dB_snr_values[:,layer_idx]

    all_null_values_dict = {
                           'spearman_r_dict':spearman_r_dict,
                           'pearson_r_dict':pearson_r_dict,
                           'dB_snr_values_dict':dB_snr_values_dict,
                           'norm_noise_dict':norm_noise_dict,
                           'norm_signal_dict':norm_signal_dict,
                           }
    
    try:
        os.makedirs(os.path.join(MODEL_DIRECTORY, PATHNULL))
    except:
        pass

    with open(pckl_path, 'wb') as handle:
        pickle.dump(all_null_values_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(raw_args=None):
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Input the arguments for the null distribution generation')
    parser.add_argument('-N', '--NUMNULL', metavar='--N', type=int, default=1000000, help='Number of samples to generate for the null distribution')
    parser.add_argument('-P', '--PATHNULL', metavar='--P', type=str, default='null_dist', help='Path to store the null distribution output. If a relative path, creates in this directory')
    parser.add_argument('-I', '--SPLITIDX', metavar='--I', type=int, default=0, help='Starting location for the null distribution reading will be (2*NUM_NULL)*SPLITIDX, so that there is no overlap between the splits. Note: for faster file reading, it can be helpful to only use even or odd splits')
    parser.add_argument('-R', '--RANDOMSEED', metavar='--R', type=int, default=0, help='random seed to use for synthesis')
    parser.add_argument('-O', '--OVERWRITE_PICKLE', metavar='--P', type=bool, default=False, help='set to true to overwrite the saved pckl file, if false then exits out if the file already exists')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='Shuffle the input loaders (both train and val). Default is to shuffle the loaders.')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false', help='Turn off shuffling for input loaders (both train and val)')
    parser.add_argument('-D', '--DIRECTORY', metavar='--D', type=str, default=None, help='The directory with the location of the `build_network.py` file. Folder structure for saving metamers will be created in this directory. If not specified, assume this script is located in the same directory as the build_network.py file.')
    parser.set_defaults(shuffle=True)

    args=parser.parse_args(raw_args)

    NUMNULL = args.NUMNULL
    SPLITIDX = args.SPLITIDX
    PATHNULL = args.PATHNULL
    RANDOMSEED = args.RANDOMSEED
    OVERWRITE_PICKLE = args.OVERWRITE_PICKLE
    shuffle = args.shuffle
    MODEL_DIRECTORY = args.DIRECTORY

    run_null_distribution(NUMNULL, SPLITIDX, PATHNULL, RANDOMSEED, OVERWRITE_PICKLE, shuffle, MODEL_DIRECTORY)

if __name__ == '__main__':
    main()
