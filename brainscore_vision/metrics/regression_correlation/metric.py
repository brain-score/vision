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
        
        if self.regression._regression.__class__ in [RidgeCV]:
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

class KernelPLSRegression:
    """PLS regression via eigendecomposition of the linear kernel.

    When n_samples < n_features, projects X into an equivalent
    (n_samples, n_samples) representation via the eigendecomposition of
    K = X_c @ X_c.T, then runs standard sklearn PLS on the reduced X.
    Mathematically identical to sklearn PLSRegression (scale=False).

    Falls back to sklearn PLSRegression when n_samples >= n_features
    or scale=True.
    """

    def __init__(self, n_components: int = 25, scale: bool = False,
                 max_iter: int = 500, tol: float = 1e-6):
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y) -> None:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n_samples, n_features = X.shape

        if n_samples >= n_features or self.scale:
            self._use_kernel = False
            self._pls = PLSRegression(n_components=self.n_components, scale=self.scale,
                                      max_iter=self.max_iter, tol=self.tol)
            self._pls.fit(X, Y)
            return

        self._use_kernel = True
        self._X_mean = X.mean(axis=0)
        X_c = X - self._X_mean
        self._X_train_centered = X_c

        K = X_c @ X_c.T
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        mask = eigenvalues > 1e-10 * eigenvalues.max()
        eigenvalues = eigenvalues[mask]
        eigenvectors = eigenvectors[:, mask]

        self._sqrt_eig = np.sqrt(eigenvalues)
        self._eigenvectors = eigenvectors
        self._inv_sqrt_eig = 1.0 / self._sqrt_eig

        X_reduced = eigenvectors * self._sqrt_eig

        n_components = min(self.n_components, X_reduced.shape[1], Y.shape[1])
        self._pls = PLSRegression(n_components=n_components, scale=False,
                                  max_iter=self.max_iter, tol=self.tol)
        self._pls.fit(X_reduced, Y)

    def predict(self, X) -> np.ndarray:
        if not self._use_kernel:
            return self._pls.predict(X)

        X = np.asarray(X, dtype=np.float64)
        X_test_c = X - self._X_mean
        K_test = X_test_c @ self._X_train_centered.T
        X_test_reduced = K_test @ (self._eigenvectors * self._inv_sqrt_eig)
        return self._pls.predict(X_test_reduced)


def pls_regression(regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = KernelPLSRegression(**regression_kwargs)
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
    regression = Ridge(**regression_kwargs)
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
    
    regression = RidgeCV(**regression_kwargs)
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
