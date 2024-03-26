from brainscore_core import Metric

from .registry import load_dataset
from brainscore_vision import load_metric
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = None
BIBTEX = """"""

pls_metric = lambda: load_metric('pls', crossvalidation_kwargs=dict(stratification_coord='object_name'))


def load_assembly(average_repetitions):
    assembly = load_dataset('SavaSegal2023-fMRI')
    assembly['region'] = 'whole_brain'
    assembly['time_bin'] = ''
    # if average_repetitions:
    #     assembly = average_repetition(assembly)
    return assembly


def _SavaSegal2023_fMRI(identifier_metric_suffix: str,
                        similarity_metric: Metric, ceiler):
    assembly_repetition = load_assembly(average_repetitions=False)
    assembly = load_assembly(average_repetitions=True)
    benchmark_identifier = f'SavaSegal2023-fMRI'
    return NeuralBenchmark(identifier=f'{benchmark_identifier}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=None,
                           bibtex=BIBTEX)

def SavaSegal2023_fMRI_pls():
    return _SavaSegal2023_fMRI('pls', pls_metric(), ceiler=lambda assembly: assembly.attrs['ceiling'])