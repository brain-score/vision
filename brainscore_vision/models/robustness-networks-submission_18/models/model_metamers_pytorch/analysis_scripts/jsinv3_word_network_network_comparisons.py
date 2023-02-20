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

import sys
sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/metamers_paper_model_analysis_folders/')
import all_model_info

sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/analysis_scripts_metamers_paper')
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

# Load the metamers for the model you are testing
layer_order, metamer_list = get_metamer_list_from_network_name(args.MET_NET, 
                                       all_model_info,
                                       network_type='audio')

prediction_dict = get_network_predictions_from_metamer_list_audio(model, metamer_list)

correct_incorrect, average_correct = evaluate_predicted_vs_actual(prediction_dict['labels_all_layers'],
                                                                  prediction_dict['predictions_all_layers'])

bootstrap_sem = bootstrap_correct_incorrect(correct_incorrect, num_bootstraps=1000)

output_dict = {'correct_incorrect':correct_incorrect,
               'average_correct':average_correct,
               'model_evaluation_directory':os.getcwd().split('/')[-1],
               'labels_all_layers':prediction_dict['labels_all_layers'],
               'predictions__all_layers':prediction_dict['predictions_all_layers'],
               'bootstrap_sem':bootstrap_sem}

with open(save_filename, 'wb') as handle:
    pckl.dump(output_dict, handle, protocol=pckl.HIGHEST_PROTOCOL)
