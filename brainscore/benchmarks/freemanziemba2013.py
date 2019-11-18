from brainscore.assemblies.private import assembly_loaders
from brainscore.benchmarks._neural_common import build_benchmark, TimeFilteredAssemblyLoader
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, TemporalCeiling
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages


def _MovshonFreemanZiemba2013Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    return build_benchmark(f'movshon.FreemanZiemba2013.private.{region}-{identifier_metric_suffix}',
                           assembly_loader=assembly_loaders[f'movshon.FreemanZiemba2013.private.{region}'],
                           similarity_metric=similarity_metric,
                           ceiler=ceiler,
                           parent=region,
                           paper_link='https://www.nature.com/articles/nn.3402')


def MovshonFreemanZiemba2013V1PLS():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V2PLS():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V1RDM():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='rdm',
                                           similarity_metric=RDMCrossValidated(
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=RDMConsistency())


def MovshonFreemanZiemba2013V2RDM():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='rdm',
                                           similarity_metric=RDMCrossValidated(
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=RDMConsistency())


def _MovshonFreemanZiemba2013TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()),
                                       crossvalidation_kwargs=dict(stratification_coord='texture_type'))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(0, 241, 10)]
    loader = assembly_loaders[f'movshon.FreemanZiemba2013.temporal.private.{region}']
    loader = TimeFilteredAssemblyLoader(loader, time_bins)
    return build_benchmark(identifier=f'movshon.FreemanZiemba2013.temporal.{region}', assembly_loader=loader,
                           similarity_metric=metric, ceiler=TemporalCeiling(InternalConsistency()),
                           parent=region, paper_link='https://www.nature.com/articles/nn.3402')


MovshonFreemanZiemba2013TemporalV1PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V1')
MovshonFreemanZiemba2013TemporalV2PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V2')
