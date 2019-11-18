from brainscore.assemblies.private import assembly_loaders
from brainscore.benchmarks._neural_common import build_benchmark, TimeFilteredAssemblyLoader
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, TemporalCeiling
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages


def _DicarloMajaj2015Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    return build_benchmark(f'dicarlo.Majaj2015.{region}-{identifier_metric_suffix}',
                           assembly_loader=assembly_loaders[f'dicarlo.Majaj2015.private.{region}'],
                           similarity_metric=similarity_metric,
                           ceiler=ceiler,
                           parent=region,
                           paper_link='http://www.jneurosci.org/content/35/39/13402.short')


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


def _DicarloMajaj2015TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = [(time_bin_start, time_bin_start + 20) for time_bin_start in range(0, 231, 20)]
    loader = TimeFilteredAssemblyLoader(assembly_loaders[f'dicarlo.Majaj2015.temporal.private.{region}'], time_bins)
    return build_benchmark(identifier=f'dicarlo.Majaj2015.temporal.{region}', assembly_loader=loader,
                           similarity_metric=metric, ceiler=TemporalCeiling(InternalConsistency()),
                           parent=region, paper_link='http://www.jneurosci.org/content/35/39/13402.short')


DicarloMajaj2015TemporalV4PLS = lambda: _DicarloMajaj2015TemporalRegion(region='V4')
DicarloMajaj2015TemporalITPLS = lambda: _DicarloMajaj2015TemporalRegion(region='IT')
