import itertools
import math
import pickle
import importlib

import brainscore

import xarray as xr
from PIL import Image
from pixelmatch.contrib.PIL import pixelmatch

from brainio.assemblies import NeuroidAssembly
from pathlib import Path

import copy
import numpy as np
import pandas as pd
from brainscore.benchmarks.screen import place_on_screen
from candidate_models.base_models import cornet

from model_tools.activations import PytorchWrapper

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from model_tools.brain_transformation import ModelCommitment
from tqdm import tqdm


SPLIT_NUMBER = 100
MAX_NUM_NEURONS = 71
HVM_TEST_IMAGES_NUM = 30
OOD_TEST_IMAGES_NUM = 30

CATEGORIES = ['apple', 'bear', 'bird', 'car', 'chair', 'dog', 'elephant', 'face', 'plane', 'zebra']
SILHOUETTE_DOMAINS = ['convex_hull', 'outline', 'skeleton', 'silhouette']

List_all_models = ['resnext101_32x16d_wsl', 'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl', 'resnext101_32x8d_wsl']


############################################################
# Loading functions: brain model specific
############################################################

def get_brainmodel(identifier, penultimate_layer:False):
    '''
    Load brain model from the correct source
    Arguments:
        identifier: Architecture name of brain model
        penultimate_layer: boolean (True: take the penultimate layer, False: take the IT layer of model,
    Returns:
        brain_model: brain model scaffold
    '''
    if identifier in ['resnet50-barlow', 'custom_model_cv_18_dagger_408', 'efficientnet-b6', 'ViT_L_32_imagenet1k', 'ViT_L_16_imagenet1k', 'r3m_resnet34', 'r3m_resnet50']:
        identifier_package_mapping = {'resnet50-barlow': 'resnet_selfsup_submission', 'custom_model_cv_18_dagger_408': 'crossvit_18_dagger_408_finetuned',
                                      'efficientnet-b6': 'efficientnet_models', 'ViT_L_32_imagenet1k': 'ViT', 'ViT_L_16_imagenet1k': 'ViT',
                                      'r3m_resnet34': 'r3m_main', 'r3m_resnet50': 'r3m_main'}
        packagename = identifier_package_mapping[identifier]
        module = importlib.import_module(f"{packagename}.models.base_models")
        get_submission_model = getattr(module, "get_model")
        get_submission_layers = getattr(module, "get_layers")
        basemodel = get_submission_model(identifier)
        layers = get_submission_layers(identifier)
        brain_model = ModelCommitment(identifier=identifier, activations_model=basemodel, layers=layers)
        return brain_model

    if (identifier == 'CORnet-S') & (penultimate_layer == True):    #TODO: Does this work?
        # only do this when choosing penultimate layer, *not* when choosing IT layer
        basemodel = cornet(identifier)
        basemodel = PytorchWrapper(model=basemodel._model, preprocessing=basemodel._extractor.preprocess)
        brain_model = ModelCommitment(identifier=identifier, activations_model=basemodel, layers=['decoder.avgpool'])
        return brain_model
    brain_model = brain_translated_pool[identifier]
    return brain_model


def retrieve_activations_from_brainmodel(brain_model, image_source: str, penultimate_layer) -> NeuroidAssembly:
    '''
    Returns a xarray DataArray with two dimensions stimulus_path and neuroid as well as the additional metadata layer on the neuroid

    Arguments:
        brain_model: Architecture name of brain model
        image_source: images for model activation
        penultimate_layer: either None if IT layer is chosen or the desired penultimate layer name

    Returns:
        activations(NeuroidAssembly xarray): activated brain model with the desired image source and layer
    '''
    # Get stimulus set for images
    stimulus_set = brainscore.get_stimulus_set(image_source)
    # Reshape images for brain model
    stimset = place_on_screen(stimulus_set, brain_model.visual_degrees(), 8)
    if penultimate_layer != None:
        # Define "recording area" in brain model
        brain_model.layer_model.region_layer_map['IT'] = penultimate_layer
    brain_model.start_recording('IT', time_bins=[(70, 170)])
    # Activate the brain model with given image dataset
    activations = brain_model.look_at(stimset)
    # Reduce to 2d array, deleting time_bin dimension
    activations = activations.squeeze()
    # Reshape the dimensions
    activations = activations.transpose('presentation', 'neuroid')

    if image_source == 'dicarlo.domain_transfer':
        # Delete unwanted sources
        activations = activations.where(activations.stimulus_source != 'GeirhosOOD', drop=True)
        activations = activations.where(activations.stimulus_source != 'CueConflict', drop=True)
        activations = activations.where(activations.stimulus_source != 'ObjectNet', drop=True)

    return activations

def get_brain_model_activation(brain_model_name, image_source, penultimate_layer_boolean=False):
    '''
    Activates brain model with respective image source and the respective layer.

    Arguments:
        brain_model_name: Architecture name of brain model
        image_source: images for model activation
        penultimate_layer: boolean (True: take the penultimate layer, False: take the IT layer of model,
    Returns:
        brain_model_activations (NeuroidAssembly xarray): activated brain model with the desired image source and layer (xarray)
    '''
    # Activate brain model with image dataset
    brain_model = get_brainmodel(brain_model_name, penultimate_layer_boolean)
    # Activate the disred layer
    if penultimate_layer_boolean:
        penultimate_layer = brain_model.layers[-1]
        brain_model_activations = retrieve_activations_from_brainmodel(brain_model, image_source, penultimate_layer)
    else:
        brain_model_activations = retrieve_activations_from_brainmodel(brain_model, image_source, penultimate_layer=None)
    return brain_model_activations

def loading_brain_model_activation(brain_model_name, image_source, penultimate_layer):   #TODO:Correct?
    '''
    Loads brain model activation and adds background ids to each of the HVM-like images (Silhouette images).

    Arguments:
        brain_model_name: Architecture name of brain model
        image_source: images for model activation
        penultimate_layer: boolean (True: take the penultimate layer, False: take the IT layer of model,
    Returns:
        domain_transfer_data
    '''
    brain_model_activation = get_brain_model_activation(brain_model_name, image_source, penultimate_layer)
    hvm_data, rest_data, non_silhouette_data = load_silhouette_data(data=brain_model_activation)
    domain_transfer_data = create_background_ids(hvm_data, rest_data, non_silhouette_data)
    return domain_transfer_data


############################################################
############################################################
############################################################


def get_brain_model_performance(brain_model_name: str, image_source:  str, estimator, image_arry, penultimate_layer: False, split_num):
    '''
    Returns a pandas dataframe with brain model performance for full image and neurons range. Performance is averaged over split_num of splits.

    Arguments:
        brain_model_name: Architecture name of brain model
        image_source: images for model activation
        estimator: Classifier for decoder
        image_arry: array with number of training images,
        penultimate_layer: boolean (True: take the penultimate layer, False: take the IT layer of model,
        split_num: number of splits to average over

    Returns:
        save pandas dataframe with full image & neuron range performance. Saves dataframe for each split and the averaged performance over all splits
        Split dataframe columns: #Neurons, #Images training, Accuracy test data
        Averaged dataframe columns: #Neurons, #Images training, Accuracy test data, Std test data
    '''
    brain_model_activations = loading_brain_model_activation(brain_model_name, image_source, penultimate_layer)
    # Calculate performance
    get_performance_splits_and_average(brain_model_activations=brain_model_activations, num_images_arry=image_arry, num_splits=split_num,
                                       estimator=estimator, brain_model_name=brain_model_name)
#    get_performance_splits_and_average_single_image(brain_model_activations=brain_model_activations, num_images=MAX_NUM_IMAGES, num_splits=SPLIT_NUMBER,
#                                                           estimator=estimator, brain_model_name=brain_model_name, num_primate_it_neurons_scaling_factor_matching=NEURONS)
    print(f'{brain_model_name} brain model performance was saved')

#################################################
#################################################
#################################################
#################################################
# Functions overlapping with hvm_crossdomain
#################################################

def create_background_ids(hvm_data, rest_data, non_silhouette_data):
    '''
    Domain-transfer data is loaded and hvm-like images get their respective background ids assigned based on their matching hvm images.

    Arguments:
        hvm_data: HVM data,
        rest_data: HVM-like (Silhouette) data,
        non_silhouette_data: non-HVM-like data (non Silhouette)

    Returns:
        domain_transfer_data: full data with an additional column: background_ids which indicates the matching background between hvm and hvm-like images
    '''
    # Add background_ids to hvm images
    hvm_data = hvm_data.assign_coords(background_id=('presentation', np.arange(1, 121)))
    non_silhouette_data = non_silhouette_data.assign_coords(
        background_id=('presentation', np.zeros(len(non_silhouette_data))))

    # Loop through each category to find the respective images
    for category in tqdm(CATEGORIES, desc='looping categories'):
        hvm_category = hvm_data[hvm_data['object_label'] == category]
        oods_category = rest_data[rest_data['object_label'] == category]
        # Find the matching backgrounds in hvm images
        background_ids = find_matching_background(oods_category, hvm_category)
        # Store the background ids in NeuronAssembly
        oods_category = oods_category.assign_coords(background_id=('presentation', background_ids))
        category_data = xr.concat((hvm_category, oods_category), dim='presentation')
        # Concate all categories together
        if category == 'apple':
            full_data = copy.copy(category_data)
        else:
            full_data = xr.concat((full_data, category_data), dim='presentation')
    domain_transfer_data = xr.concat((full_data, non_silhouette_data), dim='presentation')

    return domain_transfer_data


def find_matching_background(oods_category, hvm_category):
    '''
    hvm and hvm-like images share the same background. To identify similar backgrounds images are compared pixel-wise to each over.
    Images that share the most overlapp are then labeled with the same background id as the respective hvm-image.

    Arguments:
        oods_category (NeuronRecordingAssembly): all images from one single hvm-like domain without background id,
        hvm_category (NeuronRecordingAssembly): hvm images with background id

    Returns:
        background_ids: list of matching background ids for the single hvm-like domian
    '''
    background_ids = []
    # Find the respective background id from hvm images for each OOD image
    #'https://brainio.dicarlo.s3.amazonaws.com/assy_dicarlo_Sanghavi2021_domain_transfer.nc,8c6a02348ca892d75a83a6ffa0551e098e1edae0,dicarlo.domain_transferdicarlo.Marques2020_size,stimulus_set,StimulusSet,S3,https://brainio.dicarlo.s3.amazonaws.com/image_dicarlo_Marques2020_size.csv,0fd0aeea8fa6ff2b30ee9a6a684d4600590d631f'

    data_path = Path(__file__).parent / 'Sanghavi-domain_transfer-data/image_dicarlo_domain_transfer'
    oods_category_image_file_path = oods_category.filename

    for ood_image in tqdm(oods_category_image_file_path, desc='looping images'):
        image_filename = ood_image.item()
        image_path = str(data_path / image_filename)
        image_ood = Image.open(image_path)

        hvm_image_file_path = hvm_category.filename
        for hvm_image, hvm_background_id in zip(hvm_image_file_path, hvm_category.background_id):
            image_filename = hvm_image.item()
            image_path = str(data_path / image_filename)
            image_hvm = Image.open(image_path)
            mismatch = pixelmatch(image_hvm, image_ood)
            if mismatch <= 20000:
                background_ids.append(hvm_background_id.item())
                break
            else:
                pass
    return background_ids


def load_silhouette_data(data):
    '''
    Separating domain-transfer data into hvm, hvm-like (silhouette) and rest (non-silhouette) data. This separation is needed to give each hvm-like
    image the same background number as its respective hvm version (images are sharing the same background).

    Arguements:
        data: full data that is going to be split into hvm, hvm-like (silhouette) and rest (non-silhouette) data

    Returns:
        hvm_data (NeuronRecordingAssembly): hvm data
        rest_data (NeuronRecordingAssembly): hvm-like data
        non_silhouette_style_data (NeuronRecordingAssembly): rest data
    '''
    try:
        silhouette_style_data = data[data['identifier'] == 'Silhouette']
        non_silhouette_style_data = data[data['identifier'] != 'Silhouette']
    except:
        silhouette_style_data = data[data['stimulus_source'] == 'Silhouette']
        non_silhouette_style_data = data[data['stimulus_source'] != 'Silhouette']
    hvm_data = silhouette_style_data[silhouette_style_data['object_style'] == 'original']
    rest_data = silhouette_style_data[silhouette_style_data['object_style'] != 'original']

    return hvm_data, rest_data, non_silhouette_style_data

def get_single_domain_data(data, image_source_in_domain, object_style_in_domain):
    '''
    Filters the data for a single domain

    Arguments:
        data: NeuronRecordingAssembly xarray
        image_source_in_domain (str): Image source of wanted domain
        object_style_in_domain (str): Image style of wanted domain

    Returns:
        domain_data: data for single domain
    '''
    # Get domain data
    if image_source_in_domain in ['Art', 'Silhouette']:
        try:
            domain_data = data.where((data.identifier == image_source_in_domain) & (data.object_style == object_style_in_domain), drop=True)
        except:
            domain_data = data.where((data.stimulus_source == image_source_in_domain) & (data.object_style == object_style_in_domain), drop=True)

    else:
        try:
            domain_data = data.where(data.identifier == image_source_in_domain, drop=True)
        except:
            domain_data = data.where(data.stimulus_source == image_source_in_domain, drop=True)

    return domain_data

def get_crossdomain_data_dictionary(domain_transfer_data):
    '''
    Create a dictionary with each crossdomain data as key and its data as values

    Arguments:
        domain_transfer_data (NeuronRecordingAssembly): complete dataset

    Returns:
          dictionary with each crossdomain data as key and its data as values
    '''
    # Create dictionary
    crossdomain_data_dict = {}
    crossdomains = ['original', 'cartoon', 'line_drawing', 'mosaic', 'painting', 'sketch', 'convex_hull', 'outline', 'skeleton', 'silhouette', 'cococolor', 'cocogray', 'tdw']
    crossdomain_image_source = ['Silhouette', 'Art', 'Art', 'Art', 'Art', 'Art', 'Silhouette', 'Silhouette', 'Silhouette', 'Silhouette', 'COCOColor', 'COCOGray', 'TDW']
    for image_source, object_style in zip(crossdomain_image_source, crossdomains):
        crossdomain_data = get_single_domain_data(data=domain_transfer_data, image_source_in_domain=image_source, object_style_in_domain=object_style)
        if object_style == 'original':
            crossdomain_data_dict['hvm'] = crossdomain_data
        else:
            crossdomain_data_dict[object_style] = crossdomain_data

    return crossdomain_data_dict


def get_crossdomain_dataframes(single_neuron_image=False):
    '''
    Creates a dictionary with  each crossdomain data as key and an empty dataframe as value

    Arguments:
        single_neuron_image: boolean (True: add additional column with split number, False: no additional column)

    Returns:
        dictionary with  each crossdomain data as key and an empty dataframe as value. Columns are #Neurons, #Images training, Accuracy test data
    '''
    dataframe_dict = {}
    # Create dataframe
    if not single_neuron_image:
        df = pd.DataFrame(columns=['#Neurons', '#Images training', 'Accuracy test data'])
    else:
        df = pd.DataFrame(columns=['#Neurons', '#Images training', 'Accuracy test data', 'Split number'])

    crossdomains = ['hvm', 'cartoon', 'line_drawing', 'mosaic', 'painting', 'sketch', 'convex_hull', 'outline', 'skeleton', 'silhouette', 'cococolor', 'cocogray', 'tdw']
    for crossdomain in crossdomains:
        dataframe_dict[crossdomain] = copy.copy(df)
    return dataframe_dict

def split_training_test_images(crossdomain_data_dictionary):
    '''
    Splits data into training data pool and test images. Make sure that background id of testing hvm and training non-hvm images are not identical.

    Arguments:
        crossdomain_data_dictionary (dict): dictionary which contains each crossdomain as key and respective NeuronRecordingAssembly as values

    Returns:
        crossdomain_test_data_dictionary (dict): dictionary with each crossdomain as key and respective test images (NeuronRecordingAssembly) as values
        training_images (NeuronRecordingAssembly): training images pool. Contains only HVM images
    '''
    # Create crossdomain testing images dictionary
    crossdomain_test_data_dictionary = {}
    # Loop through each crossdomain and seed a random subset of 50 images for testing
    for crossdomain in crossdomain_data_dictionary.keys():
        crossdomain_data = crossdomain_data_dictionary[crossdomain]
        if crossdomain == 'hvm':
            test_images, training_images = reduce_data_num_images(data_complete=crossdomain_data, number_images=HVM_TEST_IMAGES_NUM)
            background_ids_silhouette_img = test_images.background_id.values

        elif crossdomain in SILHOUETTE_DOMAINS:
            test_indices = np.where(np.in1d(crossdomain_data.background_id, background_ids_silhouette_img))
            test_images = crossdomain_data[test_indices]
        else:
            test_images, _ = reduce_data_num_images(data_complete=crossdomain_data, number_images=OOD_TEST_IMAGES_NUM)
        crossdomain_test_data_dictionary[crossdomain] = test_images

    return crossdomain_test_data_dictionary, training_images

def reduce_data_num_images(data_complete, number_images):
    '''
    Draws a randomly seeded subset of data while making sure that each object category is represented equally

    Arguments:
        data_complete (NeuronRecordingAssembly): complete dataset
        number_images (int): number of images for training dataset

    Returns:
        stratified_training_data (NeuronRecordingAssembly): training data with equal number of each object category
        rest_data (NeuronRecordingAssembly): remaining data from complete data - training data
    '''
    if number_images == len(data_complete):
        place_holder = None
        return data_complete, place_holder
    else:
        try:
            stratified_training_data, rest_data = train_test_split(data_complete, train_size=number_images, stratify=data_complete.object_label)
        except:
            stratified_training_data, rest_data = train_test_split(data_complete, train_size=number_images, stratify=data_complete.category_name)

        return stratified_training_data, rest_data


def get_final_traning_data(complete_training_data, num_images_training, num_neurons):
    '''
    Draws final traning images and neurons for one split.

    Arguments:
        complete_training_data (dict with NeuronRecordingAssembly): keys: domain names, values: complete training data pool for one split,
        num_images_training: desired number of training images,
        num_neurons: desired number of training neurons

    Returns:
        final_traning_data (dict with NeuronRecordingAssembly): keys: domain names, values: final training data for this split,
        neuron_indices: training neurons indices (is used to align with the neurons in testing data)
    '''
    # Draw random subset of images from training data
    final_training_images, _ = reduce_data_num_images(data_complete=complete_training_data, number_images=num_images_training)
    # Draw random subset of neurons
    final_traning_data, neuron_indices = reduce_data_num_neurons(data=final_training_images, num_neurons=num_neurons)
    return final_traning_data, neuron_indices

def reduce_data_num_neurons(data, num_neurons):
    '''
    Reduces the number of neurons in data by randomly drawing neuron ids from complete dataset

    Arguments:
        data: (NeuronRecordingAssembly) complete dataset
        num_neurons: (int) wanted number of neurons that data shoulbe be reduced to

    Returns:
        reduced_neurons_num_data: data with reduced number of neurons
        random_indices_neurons: indices of neurons in reduced_neurons_num_data

    '''
    # Seed random numbers
    random_indices_neurons = np.random.choice(len(data.neuroid), num_neurons, replace=False)
    # Select only the random chosen neurons for training and testing data
    reduced_neurons_num_data = data[:, random_indices_neurons]
    return reduced_neurons_num_data, random_indices_neurons

def get_decoder(data, estimator):
    '''
    Trains decoder.

    Arguments:
        data: (NeuronRecordingAssembly) xarray
        estimator: (sklearn classifier function) Estimator e.g. RidgeClassifierCV, ElasticNetCV etc.

    Returns:
        clf: trained decoder
    '''
    # Get input & output data
    X = data.data
    try:
        y = data.object_label.data
    except:
        y = data.category_name.data

    # Get estimator
    clf = copy.copy(estimator) # Ridge Regression CV

    try:
        clf.fit(X, y)
    except:
        binary_label = preprocessing.LabelBinarizer()
        y = binary_label.fit_transform(y)
        clf.fit(X, y)



    return clf

def get_final_testing_data(crossdomain_test_images_dictionary, neuron_indices):
    '''
    Reduce testing data to the correct (number of) neurons.

    Arguments:
        crossdomain_test_images_dictionary (dict with NeuronRecordingAssembly): key: domain names, values: training data,
        neuron_indices: indices of desired neurons

    Returns:
        crossdomain_test_images_dictionary_final: (dict with NeuronRecordingAssembly): key: domain names, values: final training data with the correct neurons
    '''
    crossdomain_test_images_dictionary_final = {}
    for crossdomain in crossdomain_test_images_dictionary.keys():
        crossdomain_test_images_dictionary_final[crossdomain] = crossdomain_test_images_dictionary[crossdomain][:, neuron_indices]
    return crossdomain_test_images_dictionary_final

def add_accuracies_to_split_df(final_test_data_dictionary, decoder, split_dataframe, num_neurons, num_training_images):
    '''
    Fill split dataframe with decoder performance and correct number of training images and neurons that had been used in this split
    Arguments:
        final_test_data_dictionary (dict with NeuronRecordingAssembly): key: domain names, values: final training data with the correct neurons,
        decoder: trained decoder,
        split_dataframe (dict): keys: domain names, values: dataframe with columns: #Neurons, #Images training, Accuracy test data,
        num_neurons: number of training neurons,
        num_training_images: number of training images

    Retruns:
        split_dataframe (dict): keys: domain names, values: dataframe with columns: #Neurons, #Images training, Accuracy test data
        '''
    # Get and store the test accuracy for each crossdomain
    for crossdomain in final_test_data_dictionary.keys():
        test_accuracy = get_classifier_score_2AFC(classifier=decoder, data=final_test_data_dictionary[crossdomain])
        crossdomain_df = split_dataframe[crossdomain]
        # Fill dataframe
        crossdomain_df = crossdomain_df.append({
            '#Neurons': num_neurons,
            '#Images training': num_training_images,
            'Accuracy test data': test_accuracy
        }, ignore_index=True)
        split_dataframe[crossdomain] = crossdomain_df

    return split_dataframe


def get_classifier_score_2AFC(classifier, data):
    '''
    Calculates the 2AFC score

    Arguments:
        classifier: pre-trained classifier
        data (NeuronRecordingAssembly): test data

    Returns:
        2AFC score
    '''
    # Get input & output data
    X = data.data
    try:
        y = data.object_label.data
    except:
        y = data.category_name.data

    categories = np.unique(y)
    number_of_categories = len(categories)
    predict_probs = classifier.decision_function(X)
    scores = np.zeros(len(y))
    indices_row = np.arange(len(y))
    indices_column = np.arange(len(categories))

    for indx in indices_row:
        category_index = np.where(categories == y[indx])
        sum = 0
        indx_column = np.delete(indices_column, category_index)
        for idx in indx_column:
            if predict_probs[indx, category_index] > predict_probs[indx, idx]:
                sum = sum + 1
            else:
                continue

        score = sum / (number_of_categories - 1)
        scores[indx] = score

    avg_score = np.mean(scores)
    return avg_score


#################################################
#################################################
#################################################
#################################################
#################################################
# Brain model speficic functions
#################################################


def get_performance_splits_and_average(brain_model_activations, num_images_arry, num_splits, estimator, brain_model_name):
    '''Saves the real dataframes for each crossdomain and split, the fitted extrapolation parameters for each and the averaged real data performance'''

    # Check dimensionality of NeuroAssembly
    assert set(brain_model_activations.dims) == {'presentation', 'neuroid'}
    # Secure reproducibility of data
    np.random.seed(42)
    # Load data: get a dctionary with all crossdomain data
    crossdomain_data_dict = get_crossdomain_data_dictionary(domain_transfer_data=brain_model_activations)
    # Load dataframes for each crossdomain
    crossdomain_dataframes = get_crossdomain_dataframes()
    # Get the correct neuron_array for each brain_model
    num_neurons_arry = create_power_of_two_array_neurons(brain_model_activations=brain_model_activations)  #TODO: Undo the #
#    num_neurons_arry = np.asarray((1, 3, 5, 10, 20, 30, 40, 50, 71))

    # Loop through the splits
    for split in np.arange(num_splits):
        # Create in each split a new dataframe and save this one
        split_crossdomain_dataframes = get_crossdomain_dataframes()
        # Get new test images for each split, want to keep the test images consistent for one split over all images x neurons rounds
        crossdomain_test_images_dict, complete_training_data = split_training_test_images(crossdomain_data_dictionary=crossdomain_data_dict) # TODO: test if the background ids are identical for all Silhouette images in test data

        # Loop through the number of neurons
        for num_neurons, num_images_train in tqdm(itertools.product(num_neurons_arry, num_images_arry), desc='Neuron & image round'):
            # Sample final training data with the right number of neurons & images
            final_training_data, neuron_indices = get_final_traning_data(complete_training_data=complete_training_data, num_images_training=num_images_train,
                                                                         num_neurons=num_neurons)
            # Train the decoder #
            split_decoder = get_decoder(data=final_training_data, estimator=estimator)
            
            # Get the final testing data with the correct number of neurons
            final_test_data_dict = get_final_testing_data(crossdomain_test_images_dictionary=crossdomain_test_images_dict, neuron_indices=neuron_indices)
            # Get the test accuracy and store it in the split dataframe
            split_crossdomain_dataframes = add_accuracies_to_split_df(final_test_data_dictionary=final_test_data_dict, decoder=split_decoder,
                                                                      split_dataframe=split_crossdomain_dataframes, num_neurons=num_neurons, num_training_images=num_images_train)
            #TODO: correct number of training test images?
        crossdomain_dataframes = save_split_dataframes(split_crossdomain_dataframes=split_crossdomain_dataframes, crossdomain_dataframes=crossdomain_dataframes, split=split,
                                                       brain_model_name=brain_model_name)

    save_split_averaged_dataframes(crossdomain_dataframes=crossdomain_dataframes, neurons_array=num_neurons_arry, images_array=num_images_arry, brain_model_name=brain_model_name)


########################
# Other functions that are brain model specific
########################
def create_power_of_two_array_neurons(brain_model_activations):
    max_number = len(brain_model_activations.neuroid)
    # Get the potenzial for power of 2 and round the number down
    potenzial = math.floor(np.log2(max_number))
    # Create an power of two array until max number
    power_of_two_array = 2 ** np.arange(potenzial+1)

    # Add the max number of neurons to the array
    if power_of_two_array[-1] != max_number:
        power_of_two_array = np.append(power_of_two_array, max_number)

    return power_of_two_array

############################################################
# Saving functions
############################################################
def save_dataframe(dataframe, csv_dataframe_name):
    savepath = Path(__file__).parent / 'dataframes_new_models' / csv_dataframe_name #TODO: undo folder name
    dataframe.to_csv(savepath)
    print(f"Saved to {savepath}")

def save_dictionary(dictionary, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        # A new file will be created
        pickle.dump(dictionary, file)

def open_pkl(filename_pkl):
    with open(filename_pkl, 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary

####################################
# Brain model specific saving functions
####################################

def save_split_dataframes(split_crossdomain_dataframes, crossdomain_dataframes, split, brain_model_name):
#def save_split_dataframes(split_crossdomain_dataframes, crossdomain_dataframes, split, brain_model_name,primate_it_num_neurons):
    '''
    Concats each split dataframe together to get on single dataframe at the end with all performances over multiple splits and save the current split dataframe

    Arguments:
        split_crossdomain_dataframes (dict): keys: domain name, values: dataframes with performance for each #Neurons x #Images combination
        crossdomain_dataframes (dict): keys: domain name, values: dataframes with performance for each #Neurons x #Images combination stored over multiple splits
        split: number of split
        brain_model_name: name of brain model
    Returns:
        saves split dataframe for each domain
        crossdomain_dataframes (dict): keys: domain name, values: dataframes with performance for each #Neurons x #Images combination stored over multiple splits

    '''
    for crossdomain in split_crossdomain_dataframes.keys():
        crossdomain_dataframes[crossdomain] = pd.concat([crossdomain_dataframes[crossdomain], split_crossdomain_dataframes[crossdomain]], ignore_index=True)
#        save_dataframe(dataframe=split_crossdomain_dataframes[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_split_{split}_num_neurons_primate.csv')
#        save_dataframe(dataframe=split_crossdomain_dataframes[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_split_{split}_penultimate_layer.csv')
#        save_dataframe(dataframe=split_crossdomain_dataframes[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_split_{split}_penultimate_layer_multiple_neurons_{primate_it_num_neurons}_neuron_match.csv')
        save_dataframe(dataframe=split_crossdomain_dataframes[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_split_{split}_it_layer.csv')

    return crossdomain_dataframes

def save_split_averaged_dataframes(crossdomain_dataframes, neurons_array, images_array, brain_model_name):
    '''
    Saves dataframe with perfromance averaged over multiple splits for each domain.

    Arguments:
        crossdomain_dataframes (dict): keys: domain name, values: dataframes with performance for each #Neurons x #Images combination stored over multiple splits
        neurons_array: array with the number of training neurons over all splits
        images_array: array with the number of training images over all splits
        brain_model_name: name of brain model
    Returns:
         saves averaged performance dataframe for each domain
    '''
    # Average over all splits. Get mean and standard deviation
    crossdomain_dataframes_averaged = get_crossdomain_dataframes()
    for crossdomain in crossdomain_dataframes.keys():
        crossdomain_dataframes_averaged[crossdomain]['#Neurons'] = np.repeat(neurons_array, len(images_array))
        crossdomain_dataframes_averaged[crossdomain]['#Images training'] = np.tile(images_array, len(neurons_array))
        crossdomain_dataframes_averaged[crossdomain]['Accuracy test data'] = crossdomain_dataframes[crossdomain].groupby(['#Neurons', '#Images training']).mean().values
        crossdomain_dataframes_averaged[crossdomain]['Std test data'] = crossdomain_dataframes[crossdomain].groupby(['#Neurons', '#Images training']).std().values
#        save_dataframe(dataframe=crossdomain_dataframes_averaged[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_averaged_performance_num_neurons_primate.csv')
#        save_dataframe(dataframe=crossdomain_dataframes_averaged[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_averaged_performance_penultimate_layer.csv')
#        save_dataframe(dataframe=crossdomain_dataframes_averaged[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_averaged_performance_penultimate_layer_multiple_neurons.csv')
        save_dataframe(dataframe=crossdomain_dataframes_averaged[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_averaged_performance_it_layer.csv')


##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################

def get_neuron_array_for_single_img(brain_model_name, brain_model_activation, primate_it_number_of_neurons):
    brain_model_scaling_factor = get_scaling_factor_num_neurons(brain_model=brain_model_name, primate_it_num_neurons=primate_it_number_of_neurons)
    scaling_factor_multiplier = brain_model_scaling_factor/primate_it_number_of_neurons
    neuron_arry = np.asarray((10, 20, 30, 40, 50, 71))
    num_neurons_arry = neuron_arry * scaling_factor_multiplier
    round_up = np.vectorize(math.ceil)
    num_neurons_arry = round_up(num_neurons_arry)
    max_num_neurons_brain_model = len(brain_model_activation.neuroid)
    if num_neurons_arry[-1] > max_num_neurons_brain_model:
        num_neurons_arry[-1] = max_num_neurons_brain_model
    else:
        num_neurons_arry = np.append(num_neurons_arry, max_num_neurons_brain_model)

    return num_neurons_arry

def get_performance_splits_and_average_single_image(brain_model_activations, num_images, num_splits, estimator, brain_model_name, num_primate_it_neurons_scaling_factor_matching):
    '''Saves the real dataframes for each crossdomain and split, the fitted extrapolation parameters for each and the averaged real data performance'''

    # Check dimensionality of NeuroAssembly
    assert set(brain_model_activations.dims) == {'presentation', 'neuroid'}

    # Load data: get a dctionary with all crossdomain data
    crossdomain_data_dict = get_crossdomain_data_dictionary(brain_model_activations)
    # Load dataframes for each crossdomain
    crossdomain_dataframes = get_crossdomain_dataframes()
    # Get the correct neuron_array for each brain_model
    num_neurons_arry = get_neuron_array_for_single_img(brain_model_name, brain_model_activations, primate_it_number_of_neurons=num_primate_it_neurons_scaling_factor_matching)

    # Loop through  splits
    for split in np.arange(num_splits):
        # Create in each split a new dataframe and save this one
        split_crossdomain_dataframes = get_crossdomain_dataframes()
        # Get new test images for each split, want to keep the test images consistent for one split over all images x neurons rounds
        crossdomain_test_images_dict, complete_training_data = split_training_test_images(crossdomain_data_dictionary=crossdomain_data_dict)

        # Loop through the number of neurons
        for num_neurons in tqdm(num_neurons_arry, desc='Neurons'):
            # Round the number of units up
            num_neurons = math.ceil(num_neurons)
            # Sample final training data with the right number of neurons & images
            final_training_data, neuron_indices = get_final_traning_data(complete_training_data=complete_training_data, num_images_training=num_images,
                                                                         num_neurons=num_neurons)
            # Train the decoder #
            split_decoder, _ = get_decoder(data=final_training_data, estimator=estimator)
            # Get the final testing data with the correct number of neurons
            final_test_data_dict = get_final_testing_data(crossdomain_test_images_dictionary=crossdomain_test_images_dict, neuron_indices=neuron_indices)
            # Get the test accuracy and store it in the split dataframe
            split_crossdomain_dataframes = add_accuracies_to_split_df(final_test_data_dictionary=final_test_data_dict, decoder=split_decoder,
                                                                      split_dataframe=split_crossdomain_dataframes, num_neurons=num_neurons, num_training_images=num_images)

        crossdomain_dataframes = save_split_dataframes(split_crossdomain_dataframes=split_crossdomain_dataframes, crossdomain_dataframes=crossdomain_dataframes, split=split,
                                                       brain_model_name=brain_model_name, primate_it_num_neurons=num_primate_it_neurons_scaling_factor_matching)

    save_split_averaged_dataframes_single_image(crossdomain_dataframes=crossdomain_dataframes, neurons_array=num_neurons_arry, image_num=num_images, brain_model_name=brain_model_name, primate_it_num_neurons=num_primate_it_neurons_scaling_factor_matching)

def get_scaling_factor_num_neurons(brain_model, primate_it_num_neurons):
    if primate_it_num_neurons == None:
        neuron_dict = open_pkl(filename_pkl='Deep_nets_crossdomain_performance_scaling_factors_penultimate_layer.pkl')
    else:
        neuron_dict = open_pkl(filename_pkl=f'Deep_nets_crossdomain_performance_scaling_factors_penultimate_layer_{primate_it_num_neurons}_neuron_match.pkl')
    brain_model_num_neurons = neuron_dict[brain_model]
    return brain_model_num_neurons


def get_performance_splits_and_average_single_neuron_image(brain_model_activations, num_images, num_splits, estimator, brain_model_name, num_primate_it_neurons_for_scaling_factor_match):
    '''Saves the real dataframes for each crossdomain and split, the fitted extrapolation parameters for each and the averaged real data performance'''
    # Check dimensionality of NeuroAssembly
    assert set(brain_model_activations.dims) == {'presentation', 'neuroid'}

    # Load data: get a dctionary with all crossdomain data
    crossdomain_data_dict = get_crossdomain_data_dictionary(brain_model_activations)
    # Load dataframes for each crossdomain
    crossdomain_dataframes = get_crossdomain_dataframes(single_neuron_image=True)
    # Get the correct neuron_array for each brain_model
    num_neurons = get_scaling_factor_num_neurons(brain_model=brain_model_name, primate_it_num_neurons=num_primate_it_neurons_for_scaling_factor_match)

    # Loop through the kfold splits
    for split in np.arange(num_splits):
        # Get new test images for each split
        crossdomain_test_images_dict, complete_training_data = split_training_test_images(crossdomain_data_dictionary=crossdomain_data_dict)

        # Sample final training data with the right number of neurons & images
        final_training_data, neuron_indices = get_final_traning_data(complete_training_data=complete_training_data, num_images_training=num_images,
                                                                     num_neurons=num_neurons)
        # Train the decoder #
        split_decoder, _ = get_decoder(data=final_training_data, estimator=estimator)
        # Get the final testing data with the correct number of neurons
        final_test_data_dict = get_final_testing_data(crossdomain_test_images_dictionary=crossdomain_test_images_dict, neuron_indices=neuron_indices)

        # Get the test accuracy and store it in the split dataframe
        crossdomain_dataframes = add_accuracies_to_split_df_single_neuron_image(final_test_data_dictionary=final_test_data_dict, decoder=split_decoder,
                                                                  split_dataframe=crossdomain_dataframes, num_neurons=num_neurons, num_training_images=num_images,
                                                                                      split_num=split)

    save_dataframes_single_neuron_image(crossdomain_dataframes=crossdomain_dataframes, brain_model_name=brain_model_name)

def save_dataframes_single_neuron_image(crossdomain_dataframes, brain_model_name):
    for crossdomain in crossdomain_dataframes.keys():
        save_dataframe(dataframe=crossdomain_dataframes[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_scaling_factor_penultimate_layer_all_splits.csv')





def save_split_averaged_dataframes_single_image(crossdomain_dataframes, neurons_array, image_num, brain_model_name, primate_it_num_neurons):
    # Average over all splits. Get mean and standard deviation
    crossdomain_dataframes_averaged = get_crossdomain_dataframes()
    for crossdomain in crossdomain_dataframes.keys():
        crossdomain_dataframes_averaged[crossdomain]['#Neurons'] = neurons_array
        crossdomain_dataframes_averaged[crossdomain]['#Images training'] = np.repeat(image_num, repeats=len(neurons_array))
        crossdomain_dataframes_averaged[crossdomain]['Accuracy test data'] = crossdomain_dataframes[crossdomain].groupby(['#Neurons', '#Images training']).mean().values
        crossdomain_dataframes_averaged[crossdomain]['Std test data'] = crossdomain_dataframes[crossdomain].groupby(['#Neurons', '#Images training']).std().values
        save_dataframe(dataframe=crossdomain_dataframes_averaged[crossdomain], csv_dataframe_name=f'Deep_nets_performance_hvm_{crossdomain}_{brain_model_name}_averaged_performance_penultimate_layer_multiple_neurons_{primate_it_num_neurons}_neuron_match.csv')


####################################
# Data handling functions
#################################

def add_accuracies_to_split_df_single_neuron_image(final_test_data_dictionary, decoder, split_dataframe, num_neurons, num_training_images, split_num):
    # Get and store the test accuracy for each crossdomain
    for crossdomain in final_test_data_dictionary.keys():
        test_accuracy = get_classifier_score_2AFC(classifier=decoder, data=final_test_data_dictionary[crossdomain])
        crossdomain_df = split_dataframe[crossdomain]
        # Fill dataframe
        crossdomain_df = crossdomain_df.append({
            '#Neurons': num_neurons,
            '#Images training': num_training_images,
            'Accuracy test data': test_accuracy,
            'Split number': split_num
        }, ignore_index=True)
        split_dataframe[crossdomain] = crossdomain_df

    return split_dataframe



