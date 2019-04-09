import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

from brainio_base.assemblies import walk_coords
from brainscore.metrics.mask_regression import MaskRegression
from brainscore.metrics.transformations import CrossValidation
from .xarray_utils import XarrayRegression, Defaults as XarrayDefaults, XarrayCorrelation


class CrossRegressedCorrelation:
    def __init__(self, regression=None, crossvalidation_kwargs=None):
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = XarrayCorrelation(scipy.stats.pearsonr)

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class ScaledCrossRegressedCorrelation:
    def __init__(self, *args, **kwargs):
        self.cross_regressed_correlation = CrossRegressedCorrelation(*args, **kwargs)
        self.aggregate = self.cross_regressed_correlation.aggregate

    def __call__(self, source, target):
        scaled_values = scale(target, copy=True)
        target = type(target)(scaled_values, coords={
            coord: (dims, value) for coord, dims, value in walk_coords(target)}, dims=target.dims)
        return self.cross_regressed_correlation(source, target)


def mask_regression():
    regression = MaskRegression()
    regression = XarrayRegression(regression)
    return regression


def pls_regression(n_components=25, expected_dims=XarrayDefaults.expected_dims,
                   neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
    regression = PLSRegression(n_components=n_components, scale=False)
    regression = XarrayRegression(regression, expected_dims=expected_dims,
                                  neuroid_dim=neuroid_dim, neuroid_coord=neuroid_coord)
    return regression


def linear_regression(expected_dims=XarrayDefaults.expected_dims,
                      neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
    regression = LinearRegression()
    regression = XarrayRegression(regression, expected_dims=expected_dims,
                                  neuroid_dim=neuroid_dim, neuroid_coord=neuroid_coord)
    return regression
