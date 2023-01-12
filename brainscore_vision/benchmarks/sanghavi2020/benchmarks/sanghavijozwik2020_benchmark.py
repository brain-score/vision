import numpy as np

import brainscore_vision
from brainscore_vision.benchmark_helpers._neural_common import NeuralBenchmark, average_repetition
from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore_vision.utils import LazyLoad


VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 37
BIBTEX = """"""


def _DicarloSanghaviJozwik2020Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.SanghaviJozwik2020.{region}-{identifier_metric_suffix}', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloSanghaviJozwik2020V4PLS():
    return _DicarloSanghaviJozwik2020Region('V4', identifier_metric_suffix='pls',
                                            similarity_metric=CrossRegressedCorrelation(
                                                regression=pls_regression(), correlation=pearsonr_correlation(),
                                                crossvalidation_kwargs=dict(stratification_coord=None)),
                                            ceiler=InternalConsistency())


def DicarloSanghaviJozwik2020ITPLS():
    return _DicarloSanghaviJozwik2020Region('IT', identifier_metric_suffix='pls',
                                            similarity_metric=CrossRegressedCorrelation(
                                                regression=pls_regression(), correlation=pearsonr_correlation(),
                                                crossvalidation_kwargs=dict(stratification_coord=None)),
                                            ceiler=InternalConsistency())


def load_assembly(average_repetitions, region):
    assembly = brainscore_vision.load_dataset(f'dicarlo.SanghaviJozwik2020')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    assembly = assembly.sel(time_bin_id=0)  # 70-170ms
    assembly = assembly.squeeze('time_bin')
    assert NUMBER_OF_TRIALS == len(np.unique(assembly.coords['repetition']))
    assert VISUAL_DEGREES == assembly.attrs['image_size_degree']
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly
