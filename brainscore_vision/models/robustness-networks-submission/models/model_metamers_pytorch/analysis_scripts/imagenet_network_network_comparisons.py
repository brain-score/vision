"""
This file was used to get the network-network predictions by loading in the generated imagenet
metamers. Metamers are not released on the github due to file size, but this can be used 
for network-network comparisons for new networks.
"""

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
import argparse

import all_model_info

from network_network_helpers import * 

import build_network

NUM_WORKERS = 1
BATCH_SIZE = 1

#########PARSE THE ARGUMENTS FOR THE FUNCTION#########
parser = argparse.ArgumentParser(description='Input the name of the network for the metamers you want to evalalute')
parser.add_argument('MET_NET',metavar='M',type=str,help='Network from which metamers were generated.')
parser.add_argument('-F', '--FOLDER', metavar='--F', type=str, default='network_network_evaluations', help='name of the folder to save network-network evaluations')

args=parser.parse_args()

model, ds, metamer_layers = build_network.main(return_metamer_layers=True)
model.cuda()

# Path to save the evaluations
comparison_name = 'accuracy_on_metamers_from_%s.pckl'%args.MET_NET
try:
    os.makedirs(args.FOLDER)
except:
    pass
save_filename = os.path.join(args.FOLDER, comparison_name)

if os.path.isfile(save_filename):
    print('%s exists and not forcing overwrite')
    raise FileExistsError
    

# Load the metamers for the model you are testing
layer_order, metamer_list = get_metamer_list_from_network_name(args.MET_NET, 
                                       all_model_info,
                                       network_type='image')

prediction_dict = get_network_predictions_from_metamer_list_images(model, metamer_list)

correct_incorrect_1000, average_correct_1000 = evaluate_predicted_vs_actual(prediction_dict['labels_1000_way_all_layers'], 
                                                                            prediction_dict['predictions_1000_way_all_layers'])

correct_incorrect_16, average_correct_16 = evaluate_predicted_vs_actual(prediction_dict['labels_16_way_all_layers'],
                                                                        prediction_dict['predictions_16_way_all_layers'])

bootstrap_sem_1000, bootstrap_sem_16 = bootstrap_imagenet_correct_incorrect(correct_incorrect_1000, correct_incorrect_16, num_bootstraps=1000)

output_dict = {'correct_incorrect_1000':correct_incorrect_1000,
               'average_correct_1000':average_correct_1000,
               'correct_incorrect_16':correct_incorrect_16,
               'average_correct_16':average_correct_16,
               'model_evaluation_directory':os.getcwd().split('/')[-1],
               'labels_1000_way_all_layers':prediction_dict['labels_1000_way_all_layers'],
               'predictions_1000_way_all_layers':prediction_dict['predictions_1000_way_all_layers'],
               'labels_16_way_all_layers':prediction_dict['labels_16_way_all_layers'],
               'predictions_16_way_all_layers':prediction_dict['predictions_16_way_all_layers'],
               'bootstrap_sem_1000':bootstrap_sem_1000,
               'bootstrap_sem_16':bootstrap_sem_16}

with open(save_filename, 'wb') as handle:
    pckl.dump(output_dict, handle, protocol=pckl.HIGHEST_PROTOCOL)
