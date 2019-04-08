from brainscore.assemblies.private import assembly_loaders
from brainscore.benchmarks.regressing import build_benchmark
from brainscore.metrics.ceiling import InternalConsistency, TemporalCeiling
from brainscore.metrics.regression import CrossRegressedCorrelation, pearsonr_correlation, pls_regression
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages


class TimeFilteredAssemblyLoader:
    def __init__(self, baseloader, time_bins):
        self._loader = baseloader
        self._time_bins = time_bins

    def __call__(self, *args, **kwargs):
        assembly = self._loader(*args, **kwargs)
        assembly = assembly.sel(time_bin=self._time_bins)
        return assembly


def _DicarloMajaj2015TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = [(time_bin_start, time_bin_start + 20) for time_bin_start in range(0, 231, 20)]
    loader = TimeFilteredAssemblyLoader(assembly_loaders[f'dicarlo.Majaj2015.temporal.highvar.{region}'], time_bins)
    return build_benchmark(identifier=f'dicarlo.Majaj2015.temporal.{region}', assembly_loader=loader,
                           similarity_metric=metric, ceiler=TemporalCeiling(InternalConsistency()))


DicarloMajaj2015TemporalV4PLS = lambda: _DicarloMajaj2015TemporalRegion(region='V4')
DicarloMajaj2015TemporalITPLS = lambda: _DicarloMajaj2015TemporalRegion(region='IT')


def _MovshonFreemanZiemba2013TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()),
                                       crossvalidation_kwargs=dict(stratification_coord='texture_type'))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(0, 241, 10)]
    loader = assembly_loaders[f'movshon.FreemanZiemba2013.temporal.private.{region}']
    loader = TimeFilteredAssemblyLoader(loader, time_bins)
    return build_benchmark(identifier=f'movshon.FreemanZiemba2013.temporal.{region}', assembly_loader=loader,
                           similarity_metric=metric, ceiler=TemporalCeiling(InternalConsistency()))


MovshonFreemanZiemba2013TemporalV1PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V1')
MovshonFreemanZiemba2013TemporalV2PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V2')
