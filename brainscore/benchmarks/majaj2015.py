import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, TemporalCeiling
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages
from brainscore.utils import LazyLoad


def _DicarloMajaj2015Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(region))
    assembly = LazyLoad(lambda: average_repetition(assembly_repetition))
    return NeuralBenchmark(identifier=f'dicarlo.Majaj2015.{region}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region, paper_link='http://www.jneurosci.org/content/35/39/13402.short')


def DicarloMajaj2015V4PLS():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='pls',
                                   similarity_metric=CrossRegressedCorrelation(
                                       regression=pls_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def DicarloMajaj2015ITPLS():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='pls',
                                   similarity_metric=CrossRegressedCorrelation(
                                       regression=pls_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def DicarloMajaj2015V4Mask():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def DicarloMajaj2015ITMask():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='mask',
                                   similarity_metric=ScaledCrossRegressedCorrelation(
                                       regression=mask_regression(), correlation=pearsonr_correlation(),
                                       crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                   ceiler=InternalConsistency())


def DicarloMajaj2015V4RDM():
    return _DicarloMajaj2015Region('V4', identifier_metric_suffix='rdm',
                                   similarity_metric=RDMCrossValidated(
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                   ceiler=RDMConsistency())


def DicarloMajaj2015ITRDM():
    return _DicarloMajaj2015Region('IT', identifier_metric_suffix='rdm',
                                   similarity_metric=RDMCrossValidated(
                                       crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                   ceiler=RDMConsistency())


def load_assembly(region):
    assembly = brainscore.get_assembly(name='dicarlo.Majaj2015.private')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    return assembly


def _DicarloMajaj2015TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = tuple((time_bin_start, time_bin_start + 20) for time_bin_start in range(0, 231, 20))
    assembly_repetition = LazyLoad(lambda region=region, time_bins=time_bins: load_temporal_assembly(region, time_bins))
    assembly = LazyLoad(lambda: average_repetition((assembly_repetition)))
    ceiler = TemporalCeiling(InternalConsistency())
    return NeuralBenchmark(identifier=f'dicarlo.Majaj2015.temporal.{region}-pls_across_time', version=3,
                           assembly=assembly, similarity_metric=metric,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region, paper_link='http://www.jneurosci.org/content/35/39/13402.short')


DicarloMajaj2015TemporalV4PLS = lambda: _DicarloMajaj2015TemporalRegion(region='V4')
DicarloMajaj2015TemporalITPLS = lambda: _DicarloMajaj2015TemporalRegion(region='IT')


def load_temporal_assembly(region, time_bins):
    assembly = brainscore.get_assembly(name='dicarlo.Majaj2015.temporal.private')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.sel(time_bin=time_bins)
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
    return assembly
