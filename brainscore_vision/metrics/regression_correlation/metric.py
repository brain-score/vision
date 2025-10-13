import numpy as np
import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import scale

from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords, DataAssembly
from brainscore_core.metrics import Metric, Score
from brainscore_vision.metric_helpers.transformations import CrossValidation, apply_aggregate
from brainscore_vision.metric_helpers.xarray_utils import XarrayRegression, XarrayCorrelation
from brainscore_vision.metric_helpers.temporal import SpanTimeRegression, PerTime


class CrossRegressedCorrelation(Metric):
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source: DataAssembly, target: DataAssembly) -> Score:
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class ScaledCrossRegressedCorrelation(Metric):
    def __init__(self, *args, **kwargs):
        self.cross_regressed_correlation = CrossRegressedCorrelation(*args, **kwargs)
        self.aggregate = self.cross_regressed_correlation.aggregate

    def __call__(self, source: DataAssembly, target: DataAssembly) -> Score:
        scaled_values = scale(target, copy=True)
        target = target.__class__(scaled_values, coords={
            coord: (dims, value) for coord, dims, value in walk_coords(target)}, dims=target.dims)
        return self.cross_regressed_correlation(source, target)


class SingleRegression:
    def __init__(self):
        self.mapping = []

    def fit(self, X, Y):
        X = X.values
        Y = Y.values
        n_stim, n_neuroid = X.shape
        _, n_neuron = Y.shape
        r = np.zeros((n_neuron, n_neuroid))
        for neuron in range(n_neuron):
            r[neuron, :] = pearsonr(X, Y[:, neuron:neuron + 1])
        self.mapping = np.nanargmax(r, axis=1)

    def predict(self, X):
        X = X.values
        Ypred = X[:, self.mapping]
        return Ypred


# make the crc to consider time as a sample dimension
def SpanTimeCrossRegressedCorrelation(regression, correlation, *args, **kwargs):
    return CrossRegressedCorrelation(
                regression=SpanTimeRegression(regression), 
                correlation=PerTime(correlation), 
                *args, **kwargs
            )    

#fixed split metric
class FixedTrainTestSplitCorrelation(Metric):
    def __init__(self, regression, correlation, bootstrap=False):
        regression = regression or pls_regression()
        self.regression = regression
        self.correlation = correlation
        self.bootstrap = bootstrap #maybe allow bootstrapping the test set for a variance estimate in the future

    def __call__(self, source_train: DataAssembly, source_test: DataAssembly,
                target_train: DataAssembly, target_test: DataAssembly) -> Score:

        if self.bootstrap:
            raise NotImplementedError("Bootstrap not implemented yet")
            # if it was implemented, it would return: Score score with a variance
            # put the regression machinery in self.apply so that Bootstrapper could then perform multiple regressions internally
            # return metric_helpers.bootstrap.Boostrapper(source_train, target_train, source_test, target_test, self.apply, self.aggregate)
        else:
            #if no bootstrap, just do one regression on the full data
            scores = self.apply(source_train, target_train, source_test, target_test)
            score = apply_aggregate(self.aggregate, scores)    #take median across neuroids
            return score

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')

def pls_regression(regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = PLSRegression(**regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def linear_regression(xarray_kwargs=None):
    regression = LinearRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def ridge_regression(xarray_kwargs=None):
    regression = Ridge()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def single_regression(xarray_kwargs=None):
    regression = SingleRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def pearsonr_correlation(xarray_kwargs=None):
    xarray_kwargs = xarray_kwargs or {}
    return XarrayCorrelation(scipy.stats.pearsonr, **xarray_kwargs)


def pearsonr(x, y):
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True)
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True)

    r = ((xm / normxm) * (ym / normym)).sum(axis=0)

    return r
