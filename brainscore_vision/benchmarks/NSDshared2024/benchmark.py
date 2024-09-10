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
        similarity_metric: Metric, ceiler: Ceiling):
    assembly_repetition = load_assembly(f'NSD.{region}.SharedCombinedSubs.2024', average_repetitions=False)
    assembly = load_assembly(f'NSD.{region}.SharedCombinedSubs.2024', average_repetitions=True)
    benchmark_identifier = f'NSD.{region}.PLS'
    return NeuralBenchmark(identifier=f'{benchmark_identifier}-{identifier_metric_suffix}', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def NSDV1SharedPLS():
    return _NSDSharedRegion(region='V1', identifier_metric_suffix='pls',
                           similarity_metric=pls_metric(),
                           ceiler=load_ceiling('internal_consistency')) 


def load_assembly(identifier, average_repetitions=False):
    assembly = load_dataset(identifier)
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly
    
