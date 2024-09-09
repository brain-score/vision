from brainscore_core import Metric
from brainscore_core import Score

from brainscore_vision import load_metric, Ceiling, load_ceiling, load_dataset
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition

import numpy as np

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 1

BIBTEX = """@article{allen2022massive,
              title={A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence},
              author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad and Pestilli, Franco and Charest, Ian and others},
              journal={Nature neuroscience},
              volume={25},
              number={1},
              pages={116--126},
              year={2022},
              publisher={Nature Publishing Group US New York}
        }"""

pls_metric = lambda: load_metric('pls', crossvalidation_kwargs=dict(stratification_coord='object_name'))


def _NSDSharedRegion(region: str, identifier_metric_suffix: str,
                     similarity_metric: Metric):
    assembly = load_assembly(f'NSD.{region}.SharedCombinedSubs.2024')
    benchmark_identifier = f'NSD.{region}.PLS'
    # only one repetition so divide by 1
    ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    return NeuralBenchmark(identifier=f'{benchmark_identifier}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiling,
                           parent=region,
                           bibtex=BIBTEX)


def NSDV1SharedPLS():
    return _NSDSharedRegion(region='V1', identifier_metric_suffix='pls',
                           similarity_metric=pls_metric())
                           # could be used if using each subject as trial
                           #ceiler=load_ceiling('internal_consistency')) 


def load_assembly(identifier):
    assembly = load_dataset(identifier)
    # Assuming 'assembly' is your NeuronRecordingAssembly
    if 'time_bin' not in assembly.dims:
        assembly = assembly.expand_dims('time_bin').assign_coords(time_bin=[(0, 1600)])  # Add a dummy 'time_bin' with one value
    return assembly
    
