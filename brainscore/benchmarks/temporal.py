from brainscore.assemblies.private import assembly_loaders
from brainscore.benchmarks.regressing import build_benchmark
from brainscore.metrics.ceiling import InternalConsistency, TemporalCeiling
from brainscore.metrics.regression import CrossRegressedCorrelation, pearsonr_correlation, pls_regression
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages


def _DicarloMajaj2015TemporalRegion(region):
    class TimeFilteredAssemblyLoader:
        def __init__(self):
            self._loader = assembly_loaders[f'dicarlo.Majaj2015.temporal.highvar.{region}']
            self.average_repetition = self._loader.average_repetition

        def __call__(self, *args, **kwargs):
            assembly = self._loader(*args, **kwargs)
            # sub-select time-bins, and get rid of overlapping time bins
            time_bins = [(time_bin_start, time_bin_start + 20) for time_bin_start in range(0, 231, 20)]
            assembly = assembly.sel(time_bin=time_bins)
            return assembly

    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()))
    return build_benchmark(identifier=f'dicarlo.Majaj2015.temporal.{region}',
                           assembly_loader=TimeFilteredAssemblyLoader(),
                           similarity_metric=metric,
                           ceiler=TemporalCeiling(InternalConsistency()))


DicarloMajaj2015TemporalV4 = lambda: _DicarloMajaj2015TemporalRegion(region='V4')
DicarloMajaj2015TemporalIT = lambda: _DicarloMajaj2015TemporalRegion(region='IT')
