import numpy as np

from brainscore.benchmarks import BenchmarkBase, ceil_score


class NeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, similarity_metric, **kwargs):
        super(NeuralBenchmark, self).__init__(identifier=identifier, **kwargs)
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
        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return explained_variance(raw_score, self.ceiling)


def timebins_from_assembly(assembly):
    timebins = assembly['time_bin'].values
    if 'time_bin' not in assembly.dims:
        timebins = [timebins]  # only single time-bin
    return timebins


def explained_variance(score, ceiling):
    ceiled_score = ceil_score(score, ceiling)
    # ro(X, Y)
    # = (r(X, Y) / sqrt(r(X, X) * r(Y, Y)))^2
    # = (r(X, Y) / sqrt(r(Y, Y) * r(Y, Y)))^2  # assuming that r(Y, Y) ~ r(X, X) following Yamins 2014
    # = (r(X, Y) / r(Y, Y))^2
    r_square = np.power(ceiled_score.raw.sel(aggregation='center').values /
                        ceiled_score.ceiling.sel(aggregation='center').values, 2)
    ceiled_score.__setitem__({'aggregation': score['aggregation'] == 'center'}, r_square,
                             _apply_raw=False)
    return ceiled_score


def build_benchmark(identifier, assembly_loader, similarity_metric, ceiler, **kwargs):
    assembly_repetition = assembly_loader(average_repetition=False)
    assembly = assembly_loader(average_repetition=True)
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition), **kwargs)


class TimeFilteredAssemblyLoader:
    def __init__(self, baseloader, time_bins):
        self._loader = baseloader
        self._time_bins = time_bins

    def __call__(self, *args, **kwargs):
        assembly = self._loader(*args, **kwargs)
        assembly = assembly.sel(time_bin=self._time_bins)
        return assembly
