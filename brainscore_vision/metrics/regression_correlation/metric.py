import numpy as np
import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
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

class ReverseCrossRegressedCorrelation(CrossRegressedCorrelation):
    """
    Reverse predictivity version of CrossRegressedCorrelation.
    Computes neural -> model instead of model -> neural.
    """

    def __call__(self, source, target):
        # forward convention is:
        #   source = model
        #   target = neural
        # reverse predictivity swaps them
        return super().__call__(target, source)



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
class TrainTestSplitCorrelation(Metric):
    def __init__(self, regression, correlation, *args, **kwargs):
        
        regression = regression or pls_regression()
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source_train: DataAssembly, source_test: DataAssembly,
                target_train: DataAssembly, target_test: DataAssembly) -> Score:
        scores = self.apply(source_train, target_train, source_test, target_test)
        score = apply_aggregate(self.aggregate, scores)    #take median across neuroids
        return score

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        
        if hasattr(self.regression._regression, 'alpha_'):
            score.attrs['alpha'] = self.regression._regression.alpha_
            
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')

class ReverseTrainTestSplitCorrelation(TrainTestSplitCorrelation):
    """
    Reverse predictivity version of TrainTestSplitCorrelation.
    """

    def __call__(self, *, source_train, target_train, source_test, target_test):
        return super().__call__(
            source_train=target_train,
            target_train=source_train,
            source_test=target_test,
            target_test=source_test,
        )

class DualRidgeRegression:
    """Ridge regression using dual (kernel) form for memory efficiency.

    When n_samples < n_features, avoids materializing the (n_features, n_targets)
    coefficient matrix. Computes predictions via a (n_test, n_train) projection
    matrix instead. Falls back to sklearn Ridge when n_samples >= n_features.

    Mathematically identical to sklearn Ridge with fit_intercept=True.
    """

    def __init__(self, alpha: float = 1.0, chunk_size: int = 5000):
        self.alpha = alpha
        self.chunk_size = chunk_size

    def fit(self, X, Y) -> None:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n_samples, n_features = X.shape

        if n_samples >= n_features:
            self._use_dual = False
            self._primal = Ridge(alpha=self.alpha)
            self._primal.fit(X, Y)
            return

        self._use_dual = True
        self._X_mean = X.mean(axis=0)
        self._Y_mean = Y.mean(axis=0)
        X_c = X - self._X_mean
        self._X_train_centered = X_c
        self._Y_train_centered = Y - self._Y_mean

        K = X_c @ X_c.T
        K[np.diag_indices_from(K)] += self.alpha
        self._K_inv = np.linalg.solve(K, np.eye(K.shape[0]))

    def predict(self, X) -> np.ndarray:
        if not self._use_dual:
            return self._primal.predict(X)

        X = np.asarray(X, dtype=np.float64)
        X_test_c = X - self._X_mean
        proj = X_test_c @ self._X_train_centered.T @ self._K_inv

        n_test = X.shape[0]
        n_targets = self._Y_train_centered.shape[1]
        predictions = np.empty((n_test, n_targets), dtype=np.float64)
        for i in range(0, n_targets, self.chunk_size):
            end = min(i + self.chunk_size, n_targets)
            predictions[:, i:end] = proj @ self._Y_train_centered[:, i:end] + self._Y_mean[i:end]
        return predictions


class DualRidgeCVRegression:
    """RidgeCV with dual form prediction for memory efficiency.

    Uses sklearn RidgeCV for alpha selection (LOO/GCV), then the dual kernel
    form for prediction to avoid storing the (n_features, n_targets) coef_ matrix.
    Falls back to sklearn RidgeCV when n_samples >= n_features.

    Exposes ``alpha_`` after fit (selected regularization strength).
    """

    def __init__(self, alphas=None, chunk_size: int = 5000, **ridgecv_kwargs):
        self.alphas = alphas
        self.chunk_size = chunk_size
        self._ridgecv_kwargs = ridgecv_kwargs
        self.alpha_ = None

    def fit(self, X, Y) -> None:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n_samples, n_features = X.shape

        if n_samples >= n_features:
            self._use_dual = False
            self._primal = RidgeCV(alphas=self.alphas, **self._ridgecv_kwargs)
            self._primal.fit(X, Y)
            self.alpha_ = self._primal.alpha_
            return

        self._use_dual = True
        rcv = RidgeCV(alphas=self.alphas, **self._ridgecv_kwargs)
        rcv.fit(X, Y)
        self.alpha_ = rcv.alpha_
        del rcv

        self._dual = DualRidgeRegression(alpha=float(self.alpha_), chunk_size=self.chunk_size)
        self._dual.fit(X, Y)

    def predict(self, X) -> np.ndarray:
        if not self._use_dual:
            return self._primal.predict(X)
        return self._dual.predict(X)


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


def ridge_regression(regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(alpha=1)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = DualRidgeRegression(**regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression

ALPHA_LIST = [
    *[0.01, 0.1, 0.5, 1.0],
    *np.linspace(1e1, 1e2, 4, endpoint=False),
    *np.linspace(1e2, 1e3, 18, endpoint=False),
    *np.linspace(1e3, 1e4, 18, endpoint=False),
    *np.linspace(1e4, 1e5, 18, endpoint=False),
    *np.linspace(1e5, 1e6, 18, endpoint=False),
    *np.linspace(1e6, 1e7, 19)
]
def ridge_cv_regression(regression_kwargs=None, xarray_kwargs=None, alphas=ALPHA_LIST):
    regression_defaults = dict(alphas=alphas, store_cv_results=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression_kwargs.pop('alpha', None)  # RidgeCV does not accept 'alpha' as a parameter
    
    regression = DualRidgeCVRegression(**regression_kwargs)
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
