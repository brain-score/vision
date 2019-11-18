import brainscore
from brainio_base.assemblies import merge_data_arrays
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, TemporalCeiling
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages
from brainscore.utils import LazyLoad
from result_caching import store


def _MovshonFreemanZiemba2013Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(True, region=region))
    return NeuralBenchmark(identifier=f'movshon.FreemanZiemba2013.{region}-{identifier_metric_suffix}',
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition),
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


@store()
def load_assembly(average_reps, region):
    assembly = brainscore.get_assembly('movshon.FreemanZiemba2013.private')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    time_window = (50, 200)
    assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(*time_window)])
    assembly = assembly.mean(dim='time_bin', keep_attrs=True)
    assembly = assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
    assembly['time_bin_start'], assembly['time_bin_end'] = [time_window[0]], [time_window[1]]
    assembly = assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
    assembly = assembly.squeeze('time_bin')
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_reps:
        assembly = average_repetition(assembly)
    return assembly


def _MovshonFreemanZiemba2013TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()),
                                       crossvalidation_kwargs=dict(stratification_coord='texture_type'))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = tuple((time_bin_start, time_bin_start + 10) for time_bin_start in range(0, 241, 10))
    assembly_repetition = LazyLoad(lambda region=region, time_bins=time_bins:
                                   load_temporal_assembly(False, region=region, time_bins=time_bins))
    assembly = LazyLoad(lambda region=region, time_bins=time_bins:
                        load_temporal_assembly(True, region=region, time_bins=time_bins))
    ceiler = TemporalCeiling(InternalConsistency())
    return NeuralBenchmark(identifier=f'movshon.FreemanZiemba2013.temporal.{region}-pls_across_time',
                           assembly=assembly, similarity_metric=metric,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           paper_link='https://www.nature.com/articles/nn.3402')


MovshonFreemanZiemba2013TemporalV1PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V1')
MovshonFreemanZiemba2013TemporalV2PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V2')


def load_temporal_assembly(average_reps, region, time_bins):
    assembly = brainscore.get_assembly('movshon.FreemanZiemba2013.private')
    assembly = assembly.sel(region=region)
    assembly.load()
    time_assemblies = []
    for time_bin_start, time_bin_end in time_bins:
        time_assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(time_bin_start, time_bin_end)])
        time_assembly = time_assembly.mean(dim='time_bin', keep_attrs=True)
        time_assembly = time_assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
        time_assembly['time_bin_start'] = [time_bin_start]
        time_assembly['time_bin_end'] = [time_bin_end]
        time_assembly = time_assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
        time_assemblies.append(time_assembly)
    assembly = merge_data_arrays(time_assemblies)
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
    if average_reps:
        assembly = average_repetition(assembly)
    return assembly
