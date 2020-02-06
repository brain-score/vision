import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation, single_regression
from brainscore.utils import LazyLoad
from result_caching import store

VISUAL_DEGREES = 4


def _MovshonFreemanZiemba2013Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(True, region=region, access='private'))
    return NeuralBenchmark(identifier=f'movshon.FreemanZiemba2013.{region}-{identifier_metric_suffix}', version=2,
                           assembly=assembly, similarity_metric=similarity_metric, visual_degrees=VISUAL_DEGREES,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           paper_link='https://www.nature.com/articles/nn.3402')


def MovshonFreemanZiemba2013V1PLS():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V1SINGLE():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='single',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=single_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V1RDM():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='rdm',
                                           similarity_metric=RDMCrossValidated(
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=RDMConsistency())


def MovshonFreemanZiemba2013V2PLS():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V2RDM():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='rdm',
                                           similarity_metric=RDMCrossValidated(
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=RDMConsistency())


# V1 benchmarks separated by access
def _MovshonFreemanZiemba2013RegionAccess(region, identifier_metric_suffix, similarity_metric, ceiler, access='private'):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(True, region=region, access=access))
    return NeuralBenchmark(identifier=f'movshon.FreemanZiemba2013.{access}.{region}-{identifier_metric_suffix}', version=2,
                           assembly=assembly, similarity_metric=similarity_metric, visual_degrees=VISUAL_DEGREES,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           paper_link='https://www.nature.com/articles/nn.3402')


def MovshonFreemanZiemba2013V1PLSPrivate():
    return _MovshonFreemanZiemba2013RegionAccess('V1', identifier_metric_suffix='pls',
                                                 similarity_metric=CrossRegressedCorrelation(
                                                 regression=pls_regression(), correlation=pearsonr_correlation(),
                                                 crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                                 ceiler=InternalConsistency(), access='private')


def MovshonFreemanZiemba2013V1PLSPublic():
    return _MovshonFreemanZiemba2013RegionAccess('V1', identifier_metric_suffix='pls',
                                                 similarity_metric=CrossRegressedCorrelation(
                                                 regression=pls_regression(), correlation=pearsonr_correlation(),
                                                 crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                                 ceiler=InternalConsistency(), access='public')


@store()
def load_assembly(average_repetitions, region, access='private'):
    assembly = brainscore.get_assembly(f'movshon.FreemanZiemba2013.{access}')
    assembly = assembly.sel(region=region)
    assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
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
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly
