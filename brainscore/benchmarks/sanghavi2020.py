import numpy as np

import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad


VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 28
BIBTEX = """"""


def _DicarloSanghavi2020Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.Sanghavi2020.{region}-{identifier_metric_suffix}', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloSanghavi2020V4PLS():
    return _DicarloSanghavi2020Region('V4', identifier_metric_suffix='pls',
                                      similarity_metric=CrossRegressedCorrelation(
                                          regression=pls_regression(), correlation=pearsonr_correlation(),
                                          crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                      ceiler=InternalConsistency())


def DicarloSanghavi2020ITPLS():
    return _DicarloSanghavi2020Region('IT', identifier_metric_suffix='pls',
                                      similarity_metric=CrossRegressedCorrelation(
                                          regression=pls_regression(), correlation=pearsonr_correlation(),
                                          crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                      ceiler=InternalConsistency())


def load_assembly(average_repetitions, region):
    assembly = brainscore.get_assembly(name=f'dicarlo.Sanghavi2020')
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
