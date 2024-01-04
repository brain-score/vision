import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle as pckl
import os

from collections import OrderedDict
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Import the info for each of the models
from model_analysis_folders import all_model_info

def load_null_distributions(model_path,
                            null_glob,
                            layers,
                            remap_pckl_keys):
    """
    Given a model path and the name for the null files, grabs all of the null distributions. 
    Extracts the values for spearman_rho, pearson_r, dB_SNR, and raw L2 norm between the
    paired null examples. 

    Input:
        model_path (string): path to the model analysis folder
        null_glob (string): pckl file name containing the null, with wildcards for random seeds
        layers (list): list of layers to pull from the null file
        remap_pckl_keys (dict): if not None, used to make new keys in null_pckl {old_key:new_key}
    Output:
        all_nulls (dictionary): dictionary containing the distance measures for each layer
    """
    
    null_folder = glob.glob(os.path.join(model_path, null_glob))
    print('Num Null Pickles : %d'%len(null_folder))
    all_nulls = {'spearman_rho': {},
                 'pearson_r': {},
                 'dB_SNR': {}, 
                 'norm_noise': {}}
    for n_idx, null_file in enumerate(null_folder):
        loaded_null = pckl.load(open(null_file, 'rb'))
        if remap_pckl_keys is not None:
            loaded_null = remap_null_pckl_keys(loaded_null, remap_pckl_keys)
        if n_idx == 0:
            for layer in layers:
                all_nulls['spearman_rho'][layer] = np.array([]) # loaded_null['spearman_r_dict']
                all_nulls['pearson_r'][layer] = np.array([]) # loaded_null['pearson_r_dict']
                all_nulls['dB_SNR'][layer] = np.array([]) # loaded_null['dB_snr_values_dict']
                all_nulls['norm_noise'][layer] = np.array([])  # loaded_null['norm_noise_dict']
        for layer in layers:
            all_nulls['spearman_rho'][layer] = np.concatenate([all_nulls['spearman_rho'][layer], 
                                                               loaded_null['spearman_r_dict'][layer]], 0)
            all_nulls['pearson_r'][layer] = np.concatenate([all_nulls['pearson_r'][layer],
                                                               loaded_null['pearson_r_dict'][layer]], 0)
            all_nulls['dB_SNR'][layer] = np.concatenate([all_nulls['dB_SNR'][layer],
                                                               loaded_null['dB_snr_values_dict'][layer]], 0)
            all_nulls['norm_noise'][layer] = np.concatenate([all_nulls['norm_noise'][layer],
                                                               loaded_null['norm_noise_dict'][layer]], 0)

            if np.sum(loaded_null['dB_snr_values_dict'][layer] == -np.inf):
                print('Found inf values in dBSNR for file ', null_file)

    print('Total Null Measured : %d'%len(all_nulls['spearman_rho'][layers[0]]))

    return all_nulls

def remap_null_pckl_keys(loaded_null_pckl,
                         remap_pckl_keys):
    """
    Assign dictionary keys to the values in remap_pckl_keys.
    Useful if there are multiple layers with the same values but of different shapes/names.  
    """
    
    # map the old keys to the new keys
    for saved_value in list(loaded_null_pckl.keys()):
        for remap_key in list(remap_pckl_keys.keys()):
            loaded_null_pckl[saved_value][remap_pckl_keys[remap_key]] = loaded_null_pckl[saved_value][remap_key]
    
    return loaded_null_pckl    


def get_metamer_distances_wsn_word(model_path,
                                   metamer_path,
                                   layers,
                                   metamer_subset
                                   ):
    """
    Extracts the distances from the saved metamer pickle files.
    Extracts the values for spearman_rho, pearson_r, dB_SNR, and raw L2 norm between the
    metamer and the natural signal.

    Only includes metamers that pass the criteria of having the same classification on the 
    792 way word classification task. 

    Inputs:
        model_path (string): path to the model analysis folder
        metamer_path (string): subdirectory containing the saved metamers
        layers (list): list of layers to pull from each metamer pickle
        metamer_subset (list): list of metamer numerical values to include, if None include all
    """

    all_pckl_paths = glob.glob(os.path.join(model_path, metamer_path,
                                            '*_SOUND_*', 'all_metamers_pickle.pckl'))
    print('Found %d Metamers'%len(all_pckl_paths))
    if metamer_subset is not None:
        print('Only including a subset of %d in analysis'%len(metamer_subset))

    distance_measures_dict = {'spearman_rho': [],
                              'pearson_r': [],
                              'dB_SNR': [],
                              'norm_noise': []}
    all_layer_correct = []

    images_that_didnt_converge = []
    synth_images = {l:[] for l in layers}

    shape_activations_each_layer = OrderedDict()

    for m_idx, metamer_pckl in enumerate(all_pckl_paths):
        # If we are using a subset of metamers, then make sure the pckl is in the subset
        if metamer_subset is not None:
            if len([x for x in metamer_subset if '/%d_SOUND_'%x in metamer_pckl]) != 1:
                continue
        with open(metamer_pckl, 'rb') as f:
            loaded_metamer_pckl = pckl.load(f)
            distance_measures = loaded_metamer_pckl['all_distance_measures']
            distance_dict_tmp = {'spearman_rho': {},
                                 'pearson_r': {},
                                 'dB_SNR': {},
                                 'norm_noise': {}}
            layer_correct_tmp = {}
            orig_prediction = np.argmax(loaded_metamer_pckl['predictions_orig'].cpu().detach().numpy())
            for metamer_layer in layers:
                if m_idx == 0:
                    shape_activations_each_layer[metamer_layer] = loaded_metamer_pckl['all_outputs_orig'][metamer_layer].cpu().detach().numpy().shape
                synth_path_dir = os.path.dirname(metamer_pckl)
                synth_prediction = np.argmax(
                    loaded_metamer_pckl['predictions_out_dict'][metamer_layer].cpu().detach().numpy())
                layer_correct_tmp[metamer_layer] = (synth_prediction == orig_prediction)
                null_overlap_glob = glob.glob(os.path.join(synth_path_dir, 'null_overlap_images', '*_%s_synth.wav'%metamer_layer))
                if len(null_overlap_glob)==1: ## Say that the label isn't matching if we didn't pass null.
                    layer_correct_tmp[metamer_layer] = False
                assert len(null_overlap_glob)<=1, 'Found multiple null overlap images for %s'%synth_path_dir

                # images that didn't converge includes things that didn't pass the classification screen or
                # that were marked as not passing the null check
                if not (layer_correct_tmp[metamer_layer]) or (len(null_overlap_glob)==1):
                    images_that_didnt_converge.append([loaded_metamer_pckl['xadv_dict'][metamer_layer],
                        loaded_metamer_pckl['sound_orig'],
                        loaded_metamer_pckl['predictions_out_dict'][metamer_layer][0],
                        loaded_metamer_pckl['predictions_orig'][0]])
                else:
                    measured_distances = distance_measures[metamer_layer]
                    for d in distance_dict_tmp.keys():
                        distance_dict_tmp[d][metamer_layer] = {}
                    for measure_layer in measured_distances.keys():
                        for d in distance_dict_tmp.keys():
                            distance_dict_tmp[d][metamer_layer][measure_layer] = (
                                measured_distances[measure_layer][d])
                    synth_list = glob.glob(os.path.join(synth_path_dir, '*_%s_synth.wav'%metamer_layer))
                    len_assertion = 'Found %d synth.wav matching the layer when expecting 1'%len(synth_list)
                    assert(len(synth_list)==1), len_assertion
                    synth_images[metamer_layer].append(synth_list[0])

        for d in distance_dict_tmp.keys():
            distance_measures_dict[d].append(distance_dict_tmp[d])
        all_layer_correct.append(layer_correct_tmp)

    # Return as a dict so it is easy to save
    return_dict = {'distance_measures_dict': distance_measures_dict,
                   'all_layer_correct': all_layer_correct,
                   'images_that_didnt_converge': images_that_didnt_converge,
                   'synth_images': synth_images,
                   'shape_activations_each_layer': shape_activations_each_layer,
                  }
    return return_dict


def get_metamer_distances_imagenet(model_path,
                                   metamer_path,
                                   layers,
                                   metamer_subset,
                                   ):
    """
    Extracts the distances from the saved metamer pickle files.
    Extracts the values for spearman_rho, pearson_r, dB_SNR, and raw L2 norm between the
    metamer and the natural signal.

    Only includes metamers that pass the criteria of having the same classification 
    on the 16 way task as the paired natural image.

    Inputs:
        model_path (string): path to the model analysis folder
        metamer_path (string): subdirectory containing the saved metamers
        layers (list): list of layers to pull from each metamer pickle
        metamer_subset (list): list of metamer numerical values to include, if None include all
    """
    
    all_pckl_paths = glob.glob(os.path.join(model_path, metamer_path,
                                            '*_SOUND_*', 'all_metamers_pickle.pckl'))
    print('Found %d Metamers'%len(all_pckl_paths))
    if metamer_subset is not None:
        print('Only including a subset of %d in analysis'%len(metamer_subset))
    
    distance_measures_dict = {'spearman_rho': [],
                              'pearson_r': [],
                              'dB_SNR': [],
                              'norm_noise': []}
    all_layer_correct = []
    all_layer_correct_16_class = []
    
    images_that_didnt_converge = []
    synth_images = {l:[] for l in layers}

    shape_activations_each_layer = OrderedDict()
 
    for m_idx, metamer_pckl in enumerate(all_pckl_paths):
        # If we are using a subset of metamers, then make sure the pckl is in the subset
        if metamer_subset is not None:
            if len([x for x in metamer_subset if '/%d_SOUND_'%x in metamer_pckl]) != 1:
                continue
        with open(metamer_pckl, 'rb') as f:
            loaded_metamer_pckl = pckl.load(f)
            distance_measures = loaded_metamer_pckl['all_distance_measures'] 
            distance_dict_tmp = {'spearman_rho': {},
                                 'pearson_r': {},
                                 'dB_SNR': {},
                                 'norm_noise': {}}
            layer_correct_tmp = {}
            layer_correct_tmp_16_class = {}
            orig_prediction = np.argmax(loaded_metamer_pckl['predictions_orig'].cpu().detach().numpy())
            for metamer_layer in layers:
                if m_idx == 0:
                    shape_activations_each_layer[metamer_layer] = loaded_metamer_pckl['all_outputs_orig'][metamer_layer].cpu().detach().numpy().shape
                synth_path_dir = os.path.dirname(metamer_pckl)
                synth_prediction = np.argmax(
                    loaded_metamer_pckl['predictions_out_dict'][metamer_layer].cpu().detach().numpy())
                layer_correct_tmp[metamer_layer] = (synth_prediction == orig_prediction)
                layer_correct_tmp_16_class[metamer_layer] = (
                    loaded_metamer_pckl['predicted_16_cat_labels_out_dict'][metamer_layer][0] ==
                    loaded_metamer_pckl['orig_16_cat_prediction'][0])
                null_overlap_glob = glob.glob(os.path.join(synth_path_dir, 'null_overlap_images', '*_%s_synth.png'%metamer_layer))
                if len(null_overlap_glob)==1: ## Say that the label isn't matching if we didn't pass null. 
                    layer_correct_tmp_16_class[metamer_layer] = False
                    layer_correct_tmp[metamer_layer] = False
                assert len(null_overlap_glob)<=1, 'Found multiple null overlap images for %s'%synth_path_dir
               
                # images that didn't converge includes things that didn't pass the 16 way category screen or
                # that were marked as not passing the null check
                if not (layer_correct_tmp_16_class[metamer_layer]) or (len(null_overlap_glob)==1):
                    images_that_didnt_converge.append([loaded_metamer_pckl['xadv_dict'][metamer_layer],
                        loaded_metamer_pckl['sound_orig'],
                        loaded_metamer_pckl['predicted_16_cat_labels_out_dict'][metamer_layer][0],
                        loaded_metamer_pckl['orig_16_cat_prediction'][0]])
                else: 
                    measured_distances = distance_measures[metamer_layer]
                    for d in distance_dict_tmp.keys():
                        distance_dict_tmp[d][metamer_layer] = {}
                    for measure_layer in measured_distances.keys():
                        for d in distance_dict_tmp.keys():
                            distance_dict_tmp[d][metamer_layer][measure_layer] = (
                                measured_distances[measure_layer][d])
                    synth_list = glob.glob(os.path.join(synth_path_dir, '*_%s_synth.png'%metamer_layer))
                    len_assertion = 'Found %d synth.png matching the layer when expecting 1'%len(synth_list)
                    assert(len(synth_list)==1), len_assertion
                    synth_images[metamer_layer].append(synth_list[0])

        for d in distance_dict_tmp.keys():    
            distance_measures_dict[d].append(distance_dict_tmp[d])
        all_layer_correct.append(layer_correct_tmp)
        all_layer_correct_16_class.append(layer_correct_tmp_16_class)

    # Return as a dict so it is easy to save
    return_dict = {'distance_measures_dict': distance_measures_dict,
                   'all_layer_correct': all_layer_correct,
                   'all_layer_correct_16_class':all_layer_correct_16_class,
                   'images_that_didnt_converge': images_that_didnt_converge,
                   'synth_images': synth_images, 
                   'shape_activations_each_layer': shape_activations_each_layer,
                  }
    return return_dict

def print_null_dist(null, metamer, min_max_check_null_overlap='max',
                    return_overlap_images_idx=False):
    """ 
    Given the null distribution and the metamer distances for a distance metric, prints
    a table that can be used in latex checking for overlap and a second table printing the 
    median. 
    """

    if return_overlap_images_idx:
        overlap_images_idx = {}
    if min_max_check_null_overlap=='max':
        print('Distance between null and metamers')
        print('layer name & number null & number metamers & null max & metamer min \\\\')
        for i,layer in enumerate(null.keys()):
            if np.max(null[layer][:,]) > np.min(metamer[layer][:]):
                print('OVERLAP!!!')
                print(['%d:%f\n'%(idx, metamer[layer][idx]) for idx in range(len(metamer[layer][:])) if (metamer[layer][idx] < np.max(null[layer][:,]))])
                if return_overlap_images_idx:
                    overlap_images_idx[layer] = np.argwhere(metamer[layer][:] < np.max(null[layer][:,]))
                print(layer, ' & ', len(null[layer][:,]), ' & ', len(metamer[layer][:]), ' & ',
                      np.max(null[layer][:]), ' & ', 
                      np.min(metamer[layer][:]), ' \\\\')
            else:
                print(layer, ' & ', len(null[layer][:,]), ' & ', len(metamer[layer][:]), ' & ', 
                      np.max(null[layer][:]), ' & ', 
                      np.min(metamer[layer][:]), ' \\\\')
    elif min_max_check_null_overlap=='min':
        print('Distance between null and metamers')
        print('layer name & number null & number metamers & null min & metamer max \\\\')
        for i,layer in enumerate(null.keys()):
            if np.min(null[layer][:,]) < np.max(metamer[layer][:]):
                print('OVERLAP!!!')
                if return_overlap_images_idx:
                    overlap_images_idx[layer] = np.argwhere(metamer[layer][:] > np.min(null[layer][:,]))
                print(layer, ' & ', len(null[layer][:,]), ' & ', len(metamer[layer][:]), ' & ',
                      np.min(null[layer][:]), ' & ',
                      np.max(metamer[layer][:]), ' \\\\')
            else:
                print(layer, ' & ', len(null[layer][:,]), ' & ', len(metamer[layer][:]), ' & ',
                      np.min(null[layer][:]), ' & ',
                      np.max(metamer[layer][:]), ' \\\\')

    print('\n') 
    print('Median Distributions, Null and Metamers')
    print('layer name & number null & number metamers & null median & metamer median \\\\')
    for i,layer in enumerate(null.keys()):
        print(layer, ' & ', len(null[layer][:]), ' & ', len(metamer[layer][:]), ' & ', 
              np.median(null[layer][:]), ' & ', 
              np.median(metamer[layer][:]), ' \\\\')

    if return_overlap_images_idx:
        return overlap_images_idx
    else:
        return None


def print_number_of_metamers_passing_criteria_wsn_word(all_layer_correct,
                                                       return_metamer_distances=True,
                                                       distance_measure_each_metamer=None,
                                                       ):
    """
    Prints the number of metamers that passed the optimization criteria of having
    the same last activation for the wsn word task.

    If return_metamer_distances is True, returns a dictionary that contains a list
    of all of the metamer distances in distance_measure_each_metamer.
    """
    if return_metamer_distances:
        distance_list = {}

    total_correct_count = {}
    for layer_idx, layer in enumerate(all_layer_correct[0].keys()):
        for metamer_idx, metamer in enumerate(all_layer_correct):
            if metamer_idx == 0:
                if return_metamer_distances:
                    distance_list[layer] = []
                total_correct_count[layer] = 0
            total_correct_count[layer]+=bool(metamer[layer])
            if bool(metamer[layer]) and return_metamer_distances:
                distance_list[layer].append(distance_measure_each_metamer[metamer_idx][layer][layer])

    print('Total Correct Word Classification')
    print(total_correct_count)

    if return_metamer_distances:
        return distance_list


def print_number_of_metamers_passing_criteria_imagenet(all_layer_correct, 
                                                       all_layer_correct_16_class,
                                                       return_metamer_distances=True,
                                                       distance_measure_each_metamer=None,
                                                       ):
    """
    Prints the number of metamers that passed the optimization criteria of having
    the same last activation for the 1000 way task and for the 16 way task. 

    If return_metamer_distances is True, returns a dictionary that contains a list
    of all of the metamer distances in distance_measure_each_metamer. 
    """
    total_correct_count_1000_class = {}
    for layer_idx, layer in enumerate(all_layer_correct[0].keys()):
        for metamer_idx, metamer in enumerate(all_layer_correct):
            if metamer_idx == 0:
                total_correct_count_1000_class[layer] = 0
            total_correct_count_1000_class[layer]+=bool(metamer[layer])
    print('Total Correct 1000 way classification')    
    print(total_correct_count_1000_class)


    if return_metamer_distances:
        distance_list = {}
    total_correct_count_16_class = {}
    for layer_idx, layer in enumerate(all_layer_correct_16_class[0].keys()):
        for metamer_idx, metamer in enumerate(all_layer_correct_16_class):
            if metamer_idx == 0:
                if return_metamer_distances: 
                    distance_list[layer] = []
                total_correct_count_16_class[layer] = 0
            total_correct_count_16_class[layer]+=bool(metamer[layer])
            if bool(metamer[layer]) and return_metamer_distances:
                distance_list[layer].append(distance_measure_each_metamer[metamer_idx][layer][layer])
    print('Total Correct 16 way classification')            
    print(total_correct_count_16_class)
 
    if return_metamer_distances:
        return distance_list


def plot_metamer_vs_null(null, metamer_distances, plot_layers,
                         distance_label = 'Spearman rho',
                         save_plot_path=None,
                         remove_fake_relu=True):
    
    distances_for_hist = {}
    for metamer_layer_idx, metamer_layer in enumerate(plot_layers):
        distances_for_hist[metamer_layer] = {}
        for metamer_idx, metamer in enumerate(metamer_distances):
            for measure_layer_idx, measure_layer in enumerate(plot_layers):
                if metamer_idx == 0:
                    distances_for_hist[metamer_layer][measure_layer] = []
                if metamer_distances[metamer_idx].get(metamer_layer, False):
                    distances_for_hist[metamer_layer][measure_layer].append(
                        metamer_distances[metamer_idx][metamer_layer][measure_layer])
                    
    distance_min_value_by_layer = {}
    for layer in distances_for_hist.keys():
        all_metamer_layer_distances = []
        for l in distances_for_hist.keys():
            all_metamer_layer_distances += distances_for_hist[l][layer]
        distance_min_value = min(-1, min(null[layer].ravel()), 
                                 min(np.array(all_metamer_layer_distances).ravel()))
        distance_max_value = max(1, max(null[layer].ravel()), 
                                 max(np.array(all_metamer_layer_distances).ravel()))        
        if distance_max_value == np.inf:
            distance_max_value=60
        distance_min_value_by_layer[layer] = (distance_min_value, distance_max_value)
                                 
    kwargs_hist = dict(histtype='stepfilled', alpha=0.4, bins=70)
    plot_size = len(distances_for_hist.keys())
    plt.figure(figsize=(plot_size*2.5, plot_size*2.5))
    for metamer_idx, metamer_layer in enumerate(distances_for_hist.keys()):
        for measure_idx, measure_layer in enumerate(distances_for_hist.keys()): 
            if (measure_idx==0):
                metamer_weights = (np.ones(len(distances_for_hist[metamer_layer][measure_layer])) / 
                                           len(distances_for_hist[metamer_layer][measure_layer]))
                null_weights = np.ones(len(null[measure_layer]))/len(null[measure_layer])
            ax = plt.subplot(plot_size, plot_size, metamer_idx*plot_size+measure_idx+1)   
            plt.hist(distances_for_hist[metamer_layer][measure_layer], label='Metamers', color='b', 
                     weights=metamer_weights, range=distance_min_value_by_layer[measure_layer], 
                     **kwargs_hist)
            plt.hist(null[measure_layer], label='Random', color='r', 
                     weights=null_weights, range=distance_min_value_by_layer[measure_layer],
                     **kwargs_hist)
            if (metamer_idx == 0) & (measure_idx == 0):
                plt.legend(loc='upper center')
            plt.ylim(0,1)
            if metamer_idx == measure_idx:
                plt.setp(ax.spines.values(), linewidth=2)
            if (metamer_idx==0):
                if remove_fake_relu:
                    plt.title('Measure Layer: \n %s'%measure_layer.split('_fake_relu')[0], 
                              fontsize=12)
                else:
                    plt.title('Measure Layer: \n %s'%measure_layer, fontsize=12)
            if (measure_idx==0):
                if remove_fake_relu:
                    plt.ylabel('Metamer Layer: \n %s \n \n Frequency'%(
                        metamer_layer.split('_fake_relu')[0]), fontsize=12)
                else:
                    plt.ylabel('Metamer Layer: \n %s \n \n Frequency'%metamer_layer, fontsize=12)
            if (metamer_idx==(len(null.keys())-1)):
                plt.xlabel(distance_label, fontsize=12)
    plt.subplots_adjust(hspace=0.4, wspace=0.4) 
    if save_plot_path is not None:
        plt.savefig(save_plot_path, transparent=True)
        


def print_all_distance_measures_and_number_of_metamers(check_model_name,
                                                       use_saved_distances=False,
                                                       save_metamer_distances=False,
                                                       move_null_overlap_images=False, 
                                                       network_type='image',
                                                       metamer_subset=None,
                                                       check_distance_measures=None):
    if network_type=='image':
        check_model = all_model_info.ALL_NETWORKS_AND_LAYERS_IMAGES[check_model_name]
    elif network_type=='audio':
        check_model = all_model_info.ALL_NETWORKS_AND_LAYERS_AUDIO[check_model_name]
   
    # Location of where to save a single pickle containing the metamer optimization
    # criteria (rather than loading all of the invidual files)
    save_metamer_distances_location = os.path.join(check_model['location'], 
                                                   check_model['metamer_folder'],
                                                   'metamer_distances_saved.pckl')

    # If we want to use the saved file, then load it. Else load the individual
    # metamer pickles
    if use_saved_distances and os.path.isfile(save_metamer_distances_location):
        all_metamers_info = pckl.load(open(save_metamer_distances_location, 'rb'))
        loaded_metamers_file = True
    else:
        loaded_metamers_file = False
        if network_type=='image':
            all_metamers_info = get_metamer_distances_imagenet(check_model['location'], 
                                                      check_model['metamer_folder'],
                                                      check_model['layers'],
                                                      metamer_subset)
 
        elif network_type=='audio':
            all_metamers_info = get_metamer_distances_wsn_word(check_model['location'],
                                                      check_model['metamer_folder'],
                                                      check_model['layers'],
                                                      metamer_subset)
 
        if save_metamer_distances:
            pckl.dump(all_metamers_info, open(save_metamer_distances_location, 'wb'))

    # Print size of each layer
    try:
        print('Shape of Activations | Num Features')
        for l, s in all_metamers_info['shape_activations_each_layer'].items():
            print(l, " : ", s, " | ", np.prod(s))
    except KeyError:
        print('Missing Layer Shapes for this pickle')

    all_layer_correct = all_metamers_info['all_layer_correct']
    if network_type=='image':
        all_layer_correct_16_class = all_metamers_info['all_layer_correct_16_class']
    metamer_distances = all_metamers_info['distance_measures_dict']
    images_that_didnt_converge = all_metamers_info['images_that_didnt_converge']
    synth_images = all_metamers_info['synth_images']

    # Load the null distribution
    all_null_distances = load_null_distributions(check_model['location'], 
                                                 check_model['null_glob'], 
                                                 check_model['layers'],
                                                 check_model.get('remap_pckl_keys', None))
    
    if check_distance_measures is None:
        # The names to the distance measures that we will check for metamers (0) and null (1)
        all_distance_measures_metamer_and_null = [('spearman_rho', 'spearman_rho', 'max'),
                                                  ('pearson_r', 'pearson_r', 'max'),
                                                  ('dB_SNR', 'dB_SNR', 'max'),
    # The "norm_noise" distance measure (l2 distance between two images) can vary significantly 
    # depending on the base image. It is thus not reliable and we omit it here. 
    #                                               ('norm_noise', 'norm_noise', 'min'),
                                                 ]
    else:
        all_distance_measures_metamer_and_null = check_distance_measures

    plot_path = os.path.join(check_model['location'], 'plots')
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    # Now run the analyses on each of the distance measures
    for distance_pair in all_distance_measures_metamer_and_null:
        distance_measure_each_metamer = metamer_distances[distance_pair[0]]
        distance_measure_null = all_null_distances[distance_pair[1]]
        print('\n')
        print('##########################################################')
        print('####### CHECKING DISTANCE MEASURE: %s #######'%distance_pair[0])

        if network_type=='audio':       
            distance_list = print_number_of_metamers_passing_criteria_wsn_word(all_layer_correct,
                               return_metamer_distances=True,
                               distance_measure_each_metamer=distance_measure_each_metamer)
        elif network_type=='image':
            distance_list = print_number_of_metamers_passing_criteria_imagenet(all_layer_correct, 
                               all_layer_correct_16_class,
                               return_metamer_distances=True,
                               distance_measure_each_metamer=distance_measure_each_metamer)

        overlap_idxs = print_null_dist(distance_measure_null, distance_list, 
                                      min_max_check_null_overlap=distance_pair[2],
                                      return_overlap_images_idx=move_null_overlap_images)

        if move_null_overlap_images:
            print(('MOVING THE IMAGES THAT OVERLAPED WITH THE NULL DISTIRBUTION INTO '
                   'folder <image_path>/null_overlap_images. Rerun script to remake plots.'
                 ))
            for layer in overlap_idxs.keys():
                for overlap_idx_list in overlap_idxs[layer]:
                    for overlap_idx in overlap_idx_list:
                        overlap_file = synth_images[layer][overlap_idx]
                        overlap_dir= os.path.join(os.path.dirname(overlap_file), 'null_overlap_images')
                        move_file = os.path.join(overlap_dir, os.path.basename(overlap_file))
                        try:
                            os.mkdir(overlap_dir)
                        except FileExistsError:
                            pass
                        if not os.path.isfile(move_file):
                            print('Moving File %s to %s'%(overlap_file, move_file))
                            os.rename(overlap_file, move_file)

        if save_metamer_distances and not loaded_metamers_file:
            distance_hist_plot_path = os.path.join(plot_path, 
                                          'distance_hist_%s_model_%s_analysis_%s.pdf'%(
                                          distance_pair[0], check_model_name, 
                                          '_'.join(check_model['metamer_folder'].split('/'))))
        else:
            distance_hist_plot_path = None

        # Make the null vs metamer plot for the model
        plot_metamer_vs_null(distance_measure_null, distance_measure_each_metamer,
                             check_model['layers'],
                             distance_label=distance_pair[0],
                             save_plot_path=distance_hist_plot_path,
                             remove_fake_relu=True)
        plt.show()

    return synth_images

