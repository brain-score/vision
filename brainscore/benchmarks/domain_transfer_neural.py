# import general libraries
import os

# import brain-score specific libraries
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation # OODRegressedCorrelation, CrossRegressedCorrelation_with_Background
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

def _commonbase(region, identifier_metric_suffix, similarity_metric, ceiler):

    assembly_repetition = LazyLoad(lambda region=region: load_domain_transfer(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_domain_transfer(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.Sanghavi2020.domain_transfer.{region}-{identifier_metric_suffix}', version=1,
                        assembly=assembly, similarity_metric=similarity_metric,
                        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                        ceiling_func=lambda: ceiler(assembly_repetition),
                        parent=region,
                        bibtex=BIBTEX)




def IT_pls():
    return _commonbase('IT', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs = {'stratification_coord': 'object_label',
                                                                     'preprocess_indices': _preprocess_indices}), 
                                       ceiler=InternalConsistency())


#### helpers ####

def load_domain_transfer(average_repetitions, region):
    '''
     Loads the domain transfer data from local folders.

     Returns:
         assembly (NeuronRecordingAssembly): domain transfer data from pico
    '''

    assembly = brainscore.get_assembly(name=f'oleo_pico_domain_transfer')
    assembly.load()
    assembly = assembly.sel(time_bin_id=0)  # 70-170ms
    assembly = assembly.squeeze('time_bin')
    assembly = assembly.transpose('neuroid', 'presentation')

    print('Cross-Validation to only keep observations with consistency > 0.7')
    consistency = InternalConsistency()
    ceiling = consistency(assembly)
    neuroid_ceilings = ceiling.attrs['raw'].mean('split')
    good_neuroid_indices = neuroid_ceilings > 0.7
    assembly = assembly[good_neuroid_indices.values]
    assembly = assembly.dropna(dim='presentation', how='all')


    # Averaging over all repetitions
    if average_repetitions:
        assembly = average_repetition(assembly)
    
    # Delete unwanted sources
    assembly = assembly.where(assembly.stimulus_source != 'GeirhosOOD', drop=True)
    assembly = assembly.where(assembly.stimulus_source != 'CueConflict', drop=True)
    assembly = assembly.where(assembly.stimulus_source != 'ObjectNet', drop=True)
    assembly = assembly.where(assembly.object_style != 'skeleton', drop=True)
    assembly = assembly.where(assembly.object_style != 'nan', drop=True)

    ## this is temporary because i havent pushed the new version of the assembly online:
    stimulus_set = assembly.stimulus_set
    filtered_stimulus_set = stimulus_set[stimulus_set.stimulus_source != 'GeirhosOOD'].copy()
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.stimulus_source != 'CueConflict']
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.stimulus_source != 'ObjectNet']
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.object_style != 'skeleton']
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.object_style.notnull()]
    
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
        train_set = set(train_ids)
        test_set = set(test_ids)

        common_non_zero = train_set.intersection(test_set) - {0}
        if not common_non_zero:
            return train_ids, test_ids

        train_indices_to_retain = [i for i, bg_id in enumerate(train_ids) if bg_id not in common_non_zero]

        return train_indices_to_retain

def _preprocess_indices(train_indices, test_indices, source_assembly):
        train_background_ids = source_assembly['background_id'].values[train_indices]
        test_background_ids = source_assembly['background_id'].values[test_indices]
        train_indices = train_indices[get_non_overlapping_indices(train_background_ids, test_background_ids)]

        assert set(source_assembly['background_id'].values[train_indices]).intersection(set( source_assembly['background_id'].values[test_indices])) == {0}
        
        return train_indices, test_indices

