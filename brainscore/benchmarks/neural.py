import numpy as np

from brainscore.assemblies.private import assembly_loaders
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation


class NeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, similarity_metric, ceiling_func):
        super(NeuralBenchmark, self).__init__(identifier=identifier, ceiling_func=ceiling_func)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        timebins = timebins_from_assembly(self._assembly)
        self.timebins = timebins

    def __call__(self, candidate):
        candidate.start_recording(self.region, time_bins=self.timebins)
        source_assembly = candidate.look_at(self._assembly.stimulus_set)
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return ceil_score(raw_score, self.ceiling)


def timebins_from_assembly(assembly):
    timebins = assembly['time_bin'].values
    if 'time_bin' not in assembly.dims:
        timebins = [timebins]  # only single time-bin
    return timebins


def build_benchmark(identifier, assembly_loader, similarity_metric, ceiler):
    assembly_repetition = assembly_loader(average_repetition=False)
    assembly = assembly_loader(average_repetition=True)
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))


def _DicarloMajaj2015Region(region, identifier_metric_suffix, similarity_metric):
    return build_benchmark(f'dicarlo.Majaj2015.{region}-{identifier_metric_suffix}',
                           assembly_loader=assembly_loaders[f'dicarlo.Majaj2015.highvar.{region}'],
                           similarity_metric=similarity_metric,
                           ceiler=InternalConsistency())


def DicarloMajaj2015V4PLS():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='pls',
                                   similarity_metric=CrossRegressedCorrelation(
                                       regression=pls_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')))


def DicarloMajaj2015ITPLS():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='pls',
                                   similarity_metric=CrossRegressedCorrelation(
                                       regression=pls_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')))


def DicarloMajaj2015V4Mask():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')))


def DicarloMajaj2015ITMask():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')))


def DicarloMajaj2015V4RDM():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='rdm', similarity_metric=RDMCrossValidated())


def DicarloMajaj2015ITRDM():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='rdm', similarity_metric=RDMCrossValidated())


def _MovshonFreemanZiemba2013Region(region, identifier_metric_suffix, similarity_metric):
    return build_benchmark(f'movshon.FreemanZiemba2013.private.{region}-{identifier_metric_suffix}',
                           assembly_loader=assembly_loaders[f'movshon.FreemanZiemba2013.private.{region}'],
                           similarity_metric=similarity_metric,
                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V1PLS():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')))


def MovshonFreemanZiemba2013V2PLS():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')))


def ToliasCadena2017PLS():
    loader = assembly_loaders[f'tolias.Cadena2017']
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
            regression=pls_regression(), 
            correlation=pearsonr_correlation(),
            crossvalidation_kwargs={'stratification_coord': None})
    identifier = f'tolias.Cadena2017-pls'
    ceiler = InternalConsistency(split_coord='repetition_id')
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))


ToliasCadena2017 = ToliasCadena2017PLS


def ToliasCadena2017Mask():
    loader = assembly_loaders[f'tolias.Cadena2017']
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
            regression=mask_regression(), 
            correlation=pearsonr_correlation(),
            crossvalidation_kwargs={'splits': 4, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-mask'
    ceiler = InternalConsistency(split_coord='repetition_id')
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))
