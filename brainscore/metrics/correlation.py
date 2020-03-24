from scipy.stats import pearsonr

from brainscore.metrics.xarray_utils import XarrayCorrelation, Defaults as XarrayDefaults
from brainscore.metrics.transformations import TestOnlyCrossValidation


class CrossCorrelation:
    def __init__(self, stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim,
                 test_size=.8, splits=5):
        self._correlation = XarrayCorrelation(pearsonr, correlation_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        self._cross_validation = TestOnlyCrossValidation(test_size=test_size, splits=splits)
        self._neuroid_dim = neuroid_dim

    def __call__(self, source, target):
        return self._cross_validation(source, target, apply=self._correlation, aggregate=self.aggregate)

    def aggregate(self, scores):
        return scores.median(dim=self._neuroid_dim)


class Correlation:
    def __init__(self, stimulus_coord=XarrayDefaults.stimulus_coord, neuroid_coord=XarrayDefaults.neuroid_coord,
                 neuroid_dim=XarrayDefaults.neuroid_dim):
        self._correlation = XarrayCorrelation(pearsonr, correlation_coord=stimulus_coord, neuroid_coord=neuroid_coord)
        self._neuroid_dim = neuroid_dim

    def __call__(self, source, target):
        correlation = self._correlation(source, target)
        return self.aggregate(correlation)

    def aggregate(self, scores):
        return scores.median(dim=self._neuroid_dim)
