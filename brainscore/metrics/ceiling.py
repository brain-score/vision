import numpy as np
import scipy.stats
import xarray as xr

from brainscore.assemblies import walk_coords
from brainscore.metrics import Score
from brainscore.metrics.transformations import CrossValidationSingle
from brainscore.metrics.xarray_utils import Defaults as XarrayDefaults
from brainscore.metrics.xarray_utils import XarrayCorrelation


class Ceiling(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class NoCeiling(Ceiling):
    def __call__(self):
        return Score(1)


class InternalConsistency(Ceiling):
    """
    Computes the consistency within an assembly with repetitions.
    """

    class Defaults:
        split_coord = 'repetition'

    def __init__(self,
                 split_coord=Defaults.split_coord, stimulus_coord=XarrayDefaults.stimulus_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
        self._consistency = SplitHalfConsistency(stimulus_coord=stimulus_coord, neuroid_dim=neuroid_dim,
                                                 neuroid_coord=neuroid_coord)
        correction = SpearmanBrownCorrection(neuroid_dim=neuroid_dim)
        self._wrapped_consistency = self.SplitHalfWrapper(split_coord=split_coord,
                                                          consistency=self._consistency, correction=correction)
        self._cross_validation = CrossValidationSingle(train_size=0.5, split_coord=split_coord)

    def __call__(self, assembly):
        return self._cross_validation(assembly,
                                      apply=self._wrapped_consistency, aggregate=self._consistency.aggregate)

    class SplitHalfWrapper:
        def __init__(self, split_coord, consistency, correction):
            self._split_coord = split_coord
            self._consistency = consistency
            self._correction = correction

        def __call__(self, half1, half2):
            half1, half2 = self._average_repetitions(half1), self._average_repetitions(half2)
            consistency = self._consistency(half1, half2)
            consistency = self._correction(consistency, n=2)
            return consistency

        def _average_repetitions(self, assembly):
            repetition_dims = assembly[self._split_coord].dims
            nonrepetition_coords = [coord for coord, dims, values in walk_coords(assembly)
                                    if dims == repetition_dims and coord != self._split_coord]
            average = assembly.multi_groupby(nonrepetition_coords).mean(dim=repetition_dims)
            return average


class SplitHalfConsistency:
    """
    Computes the consistency between two halves of an assembly.
    """

    def __init__(self, stimulus_coord=XarrayDefaults.stimulus_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
        correlation = scipy.stats.pearsonr
        self._correlation = XarrayCorrelation(correlation, stimulus_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        self._neuroid_dim = neuroid_dim

    def __call__(self, half1, half2):
        return self._correlation(half1, half2)

    def aggregate(self, scores):
        return scores.median(dim=self._neuroid_dim)


class SpearmanBrownCorrection:
    """
    Corrects the correlation coefficients.
    """

    def __init__(self, neuroid_dim=XarrayDefaults.neuroid_dim):
        self._neuroid_dim = neuroid_dim

    def __call__(self, correlations, n):
        np.testing.assert_array_equal(correlations.dims, [self._neuroid_dim])
        return xr.apply_ufunc(lambda correlation: self.correct(correlation, n), correlations)

    def correct(self, correlation, n):
        return n * correlation / (1 + (n - 1) * correlation)


ceilings = {
    'cons': InternalConsistency,
    None: NoCeiling,
}
