# import general libraries
import os

# import brain-score specific libraries
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation # OODRegressedCorrelation, CrossRegressedCorrelation_with_Background
from brainscore.utils import LazyLoad
import brainio
from brainio.assemblies import NeuronRecordingAssembly

#### Define constants ####

VISUAL_DEGREES = 8 
NUMBER_OF_TRIALS = 63 
BIBTEX = """"""

ANIMALS = ['Oleo', 'Pico']

CATEGORIES = ['apple', 'bear', 'bird', 'car', 'chair', 'dog', 'elephant', 'face', 'plane', 'zebra']
SILHOUETTE_DOMAINS = ['convex_hull', 'outline', 'skeleton', 'silhouette']

STYLES = ['hvm', 'convex_hull', 'outline', 'silhouette', 'cartoon', 'line_drawing', 'mosaic', 'painting', 'sketch']

SPLIT_NUMBER = 100
MAX_NUM_NEURONS = 71
HVM_TEST_IMAGES_NUM = 30
OOD_TEST_IMAGES_NUM = 30



custom_cache_directory = "../work/upschrimpf1/bocini"
os.environ['RESULTCACHING_HOME'] = custom_cache_directory
os.environ['RESULTCACHING_DISABLE'] = '1'


#### Benchmark implementation ####

def _commonbase(region, identifier_metric_suffix, similarity_metric, ceiler):

    assembly_repetition = LazyLoad(lambda region=region: load_domain_transfer(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_domain_transfer(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.Sanghavi2020.{region}-{identifier_metric_suffix}', version=1,
                        assembly=assembly, similarity_metric=similarity_metric,
                        visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                        ceiling_func=lambda: ceiler(assembly_repetition),
                        parent=region,
                        bibtex=BIBTEX)


def IT_pls():
    return _commonbase('IT', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_label')), 
                                       ceiler=InternalConsistency())

#### helpers ####


def load_domain_transfer(average_repetitions, region):
    '''
     Loads the domain transfer data from local folders.

     Returns:
         assembly (NeuronRecordingAssembly): domain transfer data from pico
    '''

    # assembly = brainscore.get_assembly(name=f'dicarlo.Sanghavi2021.domain_transfer')
    assembly = brainio.assemblies.DataAssembly.from_files('/Users/ernestobocini/Desktop/brain-score-domain-transfer/packaging/domain-transfer/merged_assembly/merged_assembly.nc')
    #assembly = brainio.assemblies.DataAssembly.from_files('/work/upschrimpf1/bocini/domain-transfer/brain-score/packaging/domain_transfer/dependencies/pico_domain_transfer/assy_dicarlo_pico_domain_transfer.nc')

    assembly.load()
    assembly = assembly.sel(time_bin_id=0)  # 70-170ms
    assembly = assembly.squeeze('time_bin')
    assembly = assembly.transpose('neuroid', 'presentation')
    #assembly = NeuronRecordingAssembly(assembly)

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
    csv_path = '/Users/ernestobocini/Desktop/brain-score-domain-transfer/packaging/domain-transfer/merged_assembly/merged_stimulus_set.csv'
    dir_path = '/Users/ernestobocini/Desktop/brain-score-domain-transfer/packaging/domain-transfer/images'
    stimulus_set = brainio.stimuli.StimulusSet.from_files(csv_path, dir_path)

    #filtered_stimulus_set = assembly.stimulus_set[assembly.stimulus_set.stimulus_source != 'GeirhosOOD']
    filtered_stimulus_set = stimulus_set[stimulus_set.stimulus_source != 'GeirhosOOD'].copy()
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.stimulus_source != 'CueConflict']
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.stimulus_source != 'ObjectNet']
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.object_style != 'skeleton']
    filtered_stimulus_set = filtered_stimulus_set[filtered_stimulus_set.object_style.notnull()]
    
    # temporary, if loaded from S3, the stimulus_set is already there
    assembly.attrs['stimulus_set']=filtered_stimulus_set

    assembly = assembly.transpose('presentation', 'neuroid')

    assembly = NeuronRecordingAssembly(assembly)
    import pdb; pdb.set_trace()

    return assembly


def timebins_from_assembly(assembly):
    timebins = assembly['time_bin'].values
    if 'time_bin' not in assembly.dims:
        timebins = [timebins]  # only single time-bin
    return timebins