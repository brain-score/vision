import numpy as np

from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.loaders import assembly_loaders
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation


class NeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, similarity_metric, ceiling_func):
        super(NeuralBenchmark, self).__init__(identifier=identifier, ceiling_func=ceiling_func)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]

    def __call__(self, candidate):
        candidate.start_recording(self.region)
        source_assembly = candidate.look_at(self._assembly.stimulus_set)
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return ceil_score(raw_score, self.ceiling)


def build_benchmark(identifier, assembly_loader_name, similarity_metric, ceiler):
    loader = assembly_loaders[assembly_loader_name]
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))


def _DicarloMajaj2015Region(region, identifier_metric_suffix, similarity_metric):
    return build_benchmark(f'dicarlo.Majaj2015.{region}-{identifier_metric_suffix}',
                           assembly_loader_name=f'dicarlo.Majaj2015.highvar.{region}',
                           similarity_metric=similarity_metric,
                           ceiler=InternalConsistency(stratification_coord='object_name'))


def DicarloMajaj2015V4PLS():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='pls',
                                   similarity_metric=CrossRegressedCorrelation(
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')))


def DicarloMajaj2015ITPLS():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='pls',
                                   similarity_metric=CrossRegressedCorrelation(
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')))


def DicarloMajaj2015V4Mask():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')))


def DicarloMajaj2015ITMask():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')))


def _MovshonFreemanZiemba2013Region(region, identifier_metric_suffix, similarity_metric):
    return build_benchmark(f'movshon.FreemanZiemba2013.{region}-{identifier_metric_suffix}',
                           assembly_loader_name=f'movshon.FreemanZiemba2013.{region}',
                           similarity_metric=similarity_metric,
                           ceiler=InternalConsistency(stratification_coord=None))


def MovshonFreemanZiemba2013V1PLS():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               crossvalidation_kwargs=dict(stratification_coord=None)))


def MovshonFreemanZiemba2013V2PLS():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               crossvalidation_kwargs=dict(stratification_coord=None)))


def ToliasCadena2017():
    loader = assembly_loaders[f'tolias.Cadena2017']
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation()
    identifier = f'tolias.Cadena2017-pls'
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))
