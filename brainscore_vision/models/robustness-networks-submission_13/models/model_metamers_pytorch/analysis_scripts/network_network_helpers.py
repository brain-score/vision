import numpy as np
import pandas as pd
import os 
import glob
import math
import json
import scipy
import scipy.stats
import pickle as pckl
from PIL import Image

import model_analysis_folders.all_model_info as all_model_info

from torchvision import transforms

from analysis_scripts.helpers_16_choice import force_16_choice

from robustness.tools.label_maps import CLASS_DICT
from robustness.audio_functions import audio_transforms

import csv

from analysis_scripts.default_paths import *
from robustness.tools.audio_helpers import load_audio_wav_resample

# Get the WNID
with open(WORDNET_ID_TO_HUMAN_PATH, mode='r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    wnid_imagenet_name = {rows[0]:rows[1] for rows in reader}

# Get the word and speaker encodings for the label map
word_and_speaker_encodings = pckl.load( open(WORD_AND_SPEAKER_ENCODINGS_PATH, "rb" ))

def get_metamer_list_from_network_name(network_name, 
                                       all_model_info,
                                       network_type='image'):
    """
    Gets the list of metamers for a given network
    This assumes that the paths are present in the `metamers` directory in each 
    model folder. These are not released with the github, as the directories are 
    too large.
    """
    
    if network_type=='image':
        model_info = all_model_info.ALL_NETWORKS_AND_LAYERS_IMAGES[network_name]
    elif network_type=='audio':
        model_info = all_model_info.ALL_NETWORKS_AND_LAYERS_AUDIO[network_name]
        
    # Location of pickle containing the metamer optimization info
    metamer_distances_location = os.path.join(model_info['location'],
                                              model_info['metamer_folder'],
                                              'metamer_distances_saved.pckl')    
    
    # Load the pickle and get the list of metamers
    all_metamers_info = pckl.load(open(metamer_distances_location, 'rb'))
    print(all_metamers_info.keys())
    metamer_list = all_metamers_info['synth_images']
    layer_order = model_info['layers']
    
    return layer_order, metamer_list

def single_audio_loader(audio_path, pytorch_transforms, sr=20000, use_cuda=False):
    """Load audio and convert to pytorch input representation"""
    audio, SR = load_audio_wav_resample(audio_path, resample_SR=sr)
    # Audio transforms take in a foreground signal and a background signal
    audio, _ = pytorch_transforms(np.copy(audio), None)
    audio = audio.unsqueeze(0)
    if use_cuda:
        return audio.cuda()
    else:
        return audio

def single_image_loader(image_path, pytorch_transforms, use_cuda=False):
    """Load image and convert to pytorch input representation"""
    image = Image.open(image_path)
    image = pytorch_transforms(image).float()
    image = image.unsqueeze(0)
    if use_cuda:
        return image.cuda()
    else:
        return image
    
def get_1000_way_class_label_400_16_class_imagenet_val_images(image_path):
    """Gets the 1000-way ImageNet human-readable label for the generated metamers"""
    image_location_orig = os.path.join(ASSETS_PATH, 'full_400_16_class_imagenet_val_images/')
    image_number = image_path.split('/')[-2].split('_')[0]
    image_16_category = image_path.split('/')[-2].split('_')[-1]
    matching_images = glob.glob(os.path.join(image_location_orig, '%s_*_%s_*.JPEG'%(image_number, image_16_category)))
    assert len(matching_images) == 1
    imagenet_category = matching_images[0].split('/')[-1].split('_')[3]
    return wnid_imagenet_name[imagenet_category]


def get_network_predictions_from_metamer_list_images(model, metamer_list, 
                                                     pytorch_transforms=transforms.ToTensor()):
    """Given a pytorch model, gets the predictions for the outputs in metamer_list"""
    labels_16_way_all_layers = {}
    predictions_16_way_all_layers = {}
    labels_1000_way_all_layers = {}
    predictions_1000_way_all_layers = {}
    network_logits_all_layers = {}
    for metamer_layer in metamer_list.keys():
        network_logits = []
        predictions_16_way = []
        labels_16_way = []
        predictions_1000_way = []
        labels_1000_way = []
        for metamer_path in metamer_list[metamer_layer]:
            image = single_image_loader(metamer_path, pytorch_transforms)
            labels_16_way.append(metamer_path.split('/')[-2].split('SOUND_')[-1])
            labels_1000_way.append(get_1000_way_class_label_400_16_class_imagenet_val_images(metamer_path))
            predictions, _ = model(image.cuda(), with_latent=False) # Corresponding representation
            predictions = predictions.detach().cpu().numpy()
            network_logits.append(predictions)
            # Get the predicted 16 category label
            predictions_16_way.append(force_16_choice(np.flip(np.argsort(predictions.ravel(),0)), CLASS_DICT['ImageNet']))
            # Get the predicted 10000 category label
            predictions_1000_way.append(CLASS_DICT['ImageNet'][np.argmax(predictions)])
        labels_16_way_all_layers[metamer_layer] = labels_16_way
        predictions_16_way_all_layers[metamer_layer] = predictions_16_way
        labels_1000_way_all_layers[metamer_layer] = labels_1000_way
        predictions_1000_way_all_layers[metamer_layer] = predictions_1000_way
        network_logits_all_layers[metamer_layer] = network_logits
        
    return {'labels_16_way_all_layers': labels_16_way_all_layers, 
            'predictions_16_way_all_layers': predictions_16_way_all_layers, 
            'labels_1000_way_all_layers': labels_1000_way_all_layers, 
            'predictions_1000_way_all_layers': predictions_1000_way_all_layers, 
            'network_logits_all_layers':network_logits_all_layers,
           }

def get_network_predictions_from_metamer_list_audio(model, metamer_list,
                                                    pytorch_transforms=audio_transforms.AudioToTensor()):
    """Given a pytorch model, gets the predictions for the outputs in metamer_list"""
    labels_all_layers = {}
    predictions_all_layers = {}
    network_logits_all_layers = {}
    for metamer_layer in metamer_list.keys():
        network_logits = []
        predictions_all = []
        labels = []
        for metamer_path in metamer_list[metamer_layer]:
            # TODO: make more flexible for audio transforms? 
            audio = single_audio_loader(metamer_path, pytorch_transforms)
            labels.append(metamer_path.split('/')[-2].split('SOUND_')[-1])
            predictions, _ = model(audio.cuda(), with_latent=False) # Corresponding representation
            predictions = predictions.detach().cpu().numpy()
            network_logits.append(predictions)
            # Get the predicted 16 category label
            # Get the predicted 10000 category label
            predictions_all.append(word_and_speaker_encodings['word_idx_to_word'][np.argmax(predictions)])
        labels_all_layers[metamer_layer] = labels
        predictions_all_layers[metamer_layer] = predictions_all
        network_logits_all_layers[metamer_layer] = network_logits

    return {'labels_all_layers': labels_all_layers,
            'predictions_all_layers': predictions_all_layers,
            'network_logits_all_layers':network_logits_all_layers,
           }

def evaluate_predicted_vs_actual(labels, predictions):
    correct_incorrect = {}
    average_correct = {}
    for layer in labels.keys():
        correct_incorrect[layer] = [labels[layer][p] == predictions[layer][p] for p in range(len(predictions[layer]))]
        average_correct[layer] = np.mean(correct_incorrect[layer])
    return correct_incorrect, average_correct

def bootstrap_imagenet_correct_incorrect(correct_incorrect_1000, correct_incorrect_16, num_bootstraps=1000):
    """
    Takes the correct-incorrect predictions and bootstraps over them. 
    Run 1000 and 16 way together so that we are using the same samples.
    """
    bootstrapped_means_1000 = {}
    bootstrapped_means_16 = {}

    bootstrap_sem_1000 = {}
    bootstrap_sem_16 = {}

    for layer in correct_incorrect_1000.keys():
        bootstrapped_means_1000[layer] = []
        bootstrapped_means_16[layer] = []
        num_stimuli = len(correct_incorrect_1000[layer])
        for i in range(num_bootstraps):
            tmp_idx = np.random.choice(num_stimuli, num_stimuli)
            bootstrapped_means_1000[layer].append(np.mean(np.array(correct_incorrect_1000[layer])[tmp_idx]))
            bootstrapped_means_16[layer].append(np.mean(np.array(correct_incorrect_16[layer])[tmp_idx]))

        bootstrap_sem_1000[layer] = np.std(bootstrapped_means_1000[layer])
        bootstrap_sem_16[layer] = np.std(bootstrapped_means_16[layer])

    return bootstrap_sem_1000, bootstrap_sem_16

def bootstrap_correct_incorrect(correct_incorrect, num_bootstraps=1000):
    """
    Takes the correct-incorrect predictions and bootstraps over them.
    """
    bootstrapped_means = {}

    bootstrap_sem = {}

    for layer in correct_incorrect.keys():
        bootstrapped_means[layer] = []
        num_stimuli = len(correct_incorrect[layer])
        for i in range(num_bootstraps):
            tmp_idx = np.random.choice(num_stimuli, num_stimuli)
            bootstrapped_means[layer].append(np.mean(np.array(correct_incorrect[layer])[tmp_idx]))

        bootstrap_sem[layer] = np.std(bootstrapped_means[layer])

    return bootstrap_sem

    
