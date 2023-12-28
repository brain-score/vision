# import brain-score specific libraries
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, ridge_regression, pearsonr_correlation 
from brainscore.utils import LazyLoad
import brainscore
from brainio.assemblies import NeuronRecordingAssembly

#### Define constants ####

VISUAL_DEGREES = 8 
NUMBER_OF_TRIALS = 63 
BIBTEX = """

"""

ANIMALS = ['Oleo', 'Pico']

CATEGORIES = ['apple', 'bear', 'bird', 'car', 'chair', 'dog', 'elephant', 'face', 'plane', 'zebra']
SILHOUETTE_DOMAINS = ['convex_hull', 'outline', 'skeleton', 'silhouette']

STYLES = ['hvm', 'convex_hull', 'outline', 'silhouette', 'cartoon', 'line_drawing', 'mosaic', 'painting', 'sketch']

SPLIT_NUMBER = 100
MAX_NUM_NEURONS = 71
HVM_TEST_IMAGES_NUM = 30
OOD_TEST_IMAGES_NUM = 30


#### Benchmark implementation ####

def Igustibagus2024_ridge():
    assembly_repetition = LazyLoad(lambda: load_domain_transfer(average_repetitions=False))
    assembly = LazyLoad(lambda: load_domain_transfer(average_repetitions=True))
    similarity_metric=CrossRegressedCorrelation(
                                           regression=ridge_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs = {'stratification_coord': 'object_label',
                                                                     'preprocess_indices': _preprocess_indices})
    ceiler=InternalConsistency()
    return NeuralBenchmark(identifier=f'Igustibagus2024-pls', version=1, parent='IT',
                        assembly=assembly, similarity_metric=similarity_metric,
                        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                        ceiling_func=lambda: ceiler(assembly_repetition),
                        bibtex=BIBTEX)

#### helpers ####

def load_domain_transfer(average_repetitions):
    '''
     Loads the domain transfer data from local folders.

     Returns:
         assembly (NeuronRecordingAssembly): domain transfer data from pico
    '''

    assembly = brainscore.get_assembly(name=f'Igustibagus2024')
    assembly.load()
    assembly = assembly.sel(time_bin_id=0)  # 70-170ms
    assembly = assembly.squeeze('time_bin')
    assembly = filter_neuroids(assembly=assembly, threshold=0.7) # filter neurons with low internal consistency

    if average_repetitions:
        assembly = average_repetition(assembly) 
    
    # Delete unwanted sources from the assembly and stimulus_set
    include_sources = ['COCOColor', 'Art', 'COCOGray', 'TDW', 'Silhouette']
    include_objects = ['silhouettes', 'edges', 'original', 'sketch', 'textures', 'painting', 'mosaic', 'grayscale', 'convex_hull', 'outline', 'silhouette', 'line_drawing', 'cartoon']
    assembly = assembly.where(assembly.stimulus_source.isin(include_sources), drop=True)
    assembly = assembly.where(assembly.object_style.isin(include_objects), drop=True)
    stimulus_set = assembly.stimulus_set.copy()
    filtered_stimulus_set = stimulus_set[stimulus_set.stimulus_source.isin(include_sources)]
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.object_style.isin(include_objects)]
    # 
    assembly.attrs['stimulus_set']=filtered_stimulus_set
    assembly = assembly.transpose('presentation', 'neuroid')
    assembly = NeuronRecordingAssembly(assembly)
    
    return assembly


def timebins_from_assembly(assembly):
    timebins = assembly['time_bin'].values
    if 'time_bin' not in assembly.dims:
        timebins = [timebins]  # only single time-bin
    return timebins

def get_non_overlapping_indices(train_ids, test_ids):
        """This function identifies non-overlapping background indices between train and test sets.

        Arguments:
        train_ids -- list: IDs in the training set
        test_ids -- list: IDs in the test set

        Returns:
        train_indices_to_retain -- list: Indices of non-overlapping IDs in the training set
        """

        train_set = set(train_ids)
        test_set = set(test_ids)

        common_non_zero = train_set.intersection(test_set) - {0}
        if not common_non_zero:
            return train_ids, test_ids

        # Find indices in the training set where background IDs are not common non-zero elements
        train_indices_to_retain = [i for i, bg_id in enumerate(train_ids) if bg_id not in common_non_zero]

        return train_indices_to_retain

def _preprocess_indices(train_indices, test_indices, source_assembly):
        """This function preprocesses indices for training and testing data.

        Arguments:
        train_indices -- array-like: Candidate indices for training data
        test_indices -- array-like: Candidate indices for testing data
        source_assembly -- DataFrame: Source data containing 'background_id' column

        Returns:
        train_indices -- array-like: Preprocessed training indices
        test_indices -- array-like: Unmodified testing indices
        """

        train_background_ids = source_assembly['background_id'].values[train_indices]
        test_background_ids = source_assembly['background_id'].values[test_indices]
        train_indices = train_indices[get_non_overlapping_indices(train_background_ids, test_background_ids)]

        assert set(source_assembly['background_id'].values[train_indices]).intersection(set( source_assembly['background_id'].values[test_indices])) == {0}
        
        return train_indices, test_indices

def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    good_neuroid_indices = ceiling.attrs['raw'].mean('split') > threshold
    assembly = assembly[{'neuroid': good_neuroid_indices.values}]
    assembly = assembly.dropna(dim='presentation', how='all')
    return assembly

