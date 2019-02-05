import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from brainscore.metrics.transformations import CrossValidation
from .xarray_utils import XarrayRegression, Defaults as XarrayDefaults, XarrayCorrelation


class CrossRegressedCorrelation:
    def __init__(self):
        self.cross_validation = CrossValidation(train_size=.9)
        self.regression = pls_regression()
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
