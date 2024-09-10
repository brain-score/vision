import xarray as xr
import numpy as np

from brainscore_vision.benchmark_helpers.neural_common import Score
from brainscore_vision.metric_helpers.transformations import standard_error_of_the_mean

from .xarray_utils import apply_over_dims, recursive_op


# take the mean of scores (medians of single neuron scores) over time


def average_over_presentation(score: Score) -> Score:
    raw = score
    score = raw.mean('presentation')
    score.attrs['raw'] = raw
    return score


# PerOps is applied to every slice/chunk of the xarray along the specified dimensions
class PerOps:
    def __init__(self, callable, dims, check_coords=[]):
        # for coordinate checking, they are supposed to be the same across assemblies
        self.dims = dims
        self.callable = callable
        self.check_coords = check_coords

    def __call__(self, *asms):
        for check_coord in self.check_coords:
            asms = [asm.sortby(check_coord) for asm in asms]
            for asm in asms[1:]:
                assert (asm[check_coord].values == asms[0][check_coord].values).all()
        ret = apply_over_dims(self.callable, *asms, dims=self.dims)
        return ret


# SpanOps aggregates specified dimensions to one dimension
class SpanOps:
    def __init__(self, callable, source_dims, aggregated_dim, resample=False):
        # if resample, randomly choose samples from the aggregated dimension,
        # whose size is the same as the assembly.sizes[aggregated_dim]
        self.source_dims = source_dims
        self.aggregated_dim = aggregated_dim
        self.callable = callable
        self.resample = resample

    def __call__(self, *asms):
        asms = [self._stack(asm) for asm in asms]
        return self.callable(*asms)

    def _stack(self, assembly):
        assembly_type = type(assembly)
        size = assembly.sizes[self.aggregated_dim]
        assembly = xr.DataArray(assembly)  # xarray cannot deal with stacking MultiIndex (pydata/xarray#1554)
        assembly = assembly.reset_index(self.source_dims)
        assembly = assembly.rename({dim:dim+"_" for dim in self.source_dims})  # we'll call stacked timebins "presentation"
        assembly = assembly.stack({self.aggregated_dim : [dim+"_" for dim in self.source_dims]})
        if self.resample:
            indices = np.random.randint(0, assembly.sizes[self.aggregated_dim], size)
            assembly = assembly.isel({self.aggregated_dim: indices})
        return assembly_type(assembly)

class PerTime(PerOps):
    def __init__(self, callable, time_dim="time_bin", check_coord="time_bin_start", **kwargs):
        self.time_bin = time_dim
        super().__init__(callable, dims=[time_dim], check_coords=[check_coord], **kwargs)
 
class PerPresentation(PerOps):
    def __init__(self, callable, presentation_dim="presentation", check_coord="stimulus_id", **kwargs):
        self.presentation_dim = presentation_dim
        super().__init__(callable, dims=[presentation_dim], check_coords=[check_coord], **kwargs)

class PerNeuroid(PerOps):
    def __init__(self, callable, neuroid_dim="neuroid", check_coord="neuroid_id", **kwargs):
        self.neuroid_dim = neuroid_dim
        super().__init__(callable, dims=[neuroid_dim], check_coords=[check_coord], **kwargs)

class SpanTime(SpanOps):
    def __init__(self, callable, time_dim="time_bin", presentation_dim="presentation", resample=False):
        self.time_dim = time_dim
        self.presentation_dim = presentation_dim
        source_dims = [self.time_dim, self.presentation_dim]
        aggregated_dim = self.presentation_dim
        super().__init__(callable, source_dims, aggregated_dim, resample=resample)

class SpanTimeRegression:
    """
    Fits a regression with weights shared across the time bins.
    """

    def __init__(self, regression):
        self._regression = regression

    def fit(self, source, target):
        assert (source['time_bin'].values == target['time_bin'].values).all()
        SpanTime(self._regression.fit)(source, target)

    def predict(self, source):
        return PerTime(self._regression.predict)(source)

class PerTimeRegression:
    """
    Fits a regression with different weights for each time bins.
    """

    def __init__(self, regression):
        self._regression = regression

    def fit(self, source, target):
        # Lazy fit until predict
        assert (source['time_bin'].values == target['time_bin'].values).all()
        self._train_source = source
        self._train_target = target

    def predict(self, source):
        def fit_predict(train_source, train_target, test_source):
            self._regression.fit(train_source, train_target)
            return self._regression.predict(test_source)
        return PerTime(fit_predict)(self._train_source, self._train_target, source)