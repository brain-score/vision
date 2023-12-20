# import general libraries
import numpy as np
import pandas as pd
import pickle
import copy
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split


# import brain-score specific libraries
from brainscore.utils import LazyLoad
from brainscore.benchmarks import BenchmarkBase
from brainio.fetch import get_stimulus_set
from brainscore.model_interface import BrainModel
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score, accuracy



#### define constants ####
VISUAL_DEGREES = 8 
NUMBER_OF_TRIALS = 63 
CATEGORIES = ['apple', 'bear', 'bird', 'car', 'chair', 'dog', 'elephant', 'face', 'plane', 'zebra']
SILHOUETTE_DOMAINS = ['convex_hull', 'outline', 'skeleton', 'silhouette']
HVM_TEST_IMAGES_NUM = 30
OOD_TEST_IMAGES_NUM = 30
NUM_SPLITS = 1000



BIBTEX = """
@inproceedings{bagus2022primate,
  title={Primate inferotemporal cortex neurons generalize better to novel image distributions than analogous deep neural networks units},
  author={Bagus, Ayu Marliawaty I Gusti and Marques, Tiago and Sanghavi, Sachi and DiCarlo, James J and Schrimpf, Martin},
  booktitle={SVRHM 2022 Workshop@ NeurIPS},
  year={2022}
}
"""


#### Analysis Benchmark Implementation  ####

class _OOD_AnalysisBenchmark(BenchmarkBase):
    def __init__(self, classifier):

        self._classifier= classifier
        self._fitting_stimuli = get_stimulus_set('Igustibagus2024')
        self._fitting_stimuli.identifier = 'domain_transfer_pico_oleo'
        self._visual_degrees = 8
        super(_OOD_AnalysisBenchmark, self).__init__(
            identifier='dicarlo.OOD_Analysis_Benchmark',
            version=1,
            ceiling_func=lambda: self._classifier.ceiling(self._assembly),
            parent='behavior',
            bibtex=BIBTEX,
        )
    # The __call__ method takes as input a candidate BrainModel and outputs a similarity score of how brain-like
    # the candidate is under this benchmark.
    # A candidate here could be a model such as CORnet or brain-mapped Alexnet, but importantly the benchmark can be
    # agnostic to the details of the candidate and instead only engage with the BrainModel interface.
    def __call__(self, candidate: BrainModel) -> Score:
        # based on the visual degrees of the candidate
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        candidate.start_recording('IT', time_bins=[(70, 170)])
        activations = candidate.look_at(fitting_stimuli)
        activations = activations.squeeze()
        activations = activations.transpose('presentation', 'neuroid')

        crossdomain_data_dict = get_crossdomain_data_dictionary(domain_transfer_data=activations)
        
        crossdomain_results = [] # {}
        for split in tqdm(np.arange(NUM_SPLITS)):
            crossdomain_test_images_dict, complete_training_data = split_training_test_images(crossdomain_data_dictionary=crossdomain_data_dict)

            # Train the decoder #
            decoder = get_decoder(data=complete_training_data, estimator=self._classifier)
            for crossdomain in crossdomain_test_images_dict.keys():
                test_accuracy = get_classifier_score_2AFC(classifier=decoder, data=crossdomain_test_images_dict[crossdomain])
                test_accuracy = Score(test_accuracy)
                test_accuracy = test_accuracy.expand_dims('domain').expand_dims('split')
                test_accuracy['domain'] = [crossdomain]
                test_accuracy['split'] = [split]
                crossdomain_results.append(test_accuracy) 

        crossdomain_results = Score.merge(*crossdomain_results)
        crossdomain_results = crossdomain_results.mean(dim='split')
            
        return crossdomain_results
    

def OOD_AnalysisBenchmark():
    return _OOD_AnalysisBenchmark(
        classifier=RidgeClassifierCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], fit_intercept=True, normalize=True) 
    )

########## helpers #############
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
            test_images, training_images =reduce_data_num_images(data_complete=crossdomain_data, number_images=HVM_TEST_IMAGES_NUM)
            background_ids_silhouette_img = test_images.background_id.values

        elif crossdomain in SILHOUETTE_DOMAINS:
            test_indices = np.where(np.in1d(crossdomain_data.background_id, background_ids_silhouette_img))
            test_images = crossdomain_data[test_indices]
        else:
            test_images, _ = reduce_data_num_images(data_complete=crossdomain_data, number_images=OOD_TEST_IMAGES_NUM)
        crossdomain_test_data_dictionary[crossdomain] = test_images

    return crossdomain_test_data_dictionary, training_images


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