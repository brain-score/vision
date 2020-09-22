import numpy as np

import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad


VISUAL_DEGREES = 5
NUMBER_OF_TRIALS = 50
BIBTEX = """"""


def _DicarloSanghaviMurty2020Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.SanghaviMurty2020.{region}-{identifier_metric_suffix}', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloSanghaviMurty2020V4PLS():
    return _DicarloSanghaviMurty2020Region('V4', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord=None)),
                                           ceiler=InternalConsistency())


def DicarloSanghaviMurty2020ITPLS():
    return _DicarloSanghaviMurty2020Region('IT', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord=None)),
                                           ceiler=InternalConsistency())


def load_assembly(average_repetitions, region):
    assembly = brainscore.get_assembly(name=f'dicarlo.SanghaviMurty2020')
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
