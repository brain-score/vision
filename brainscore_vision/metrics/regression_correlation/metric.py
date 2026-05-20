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

    When n_samples < n_features, uses adaptive storage to minimize memory
    after fit:
    - If n_targets < n_samples: computes coef_ and frees X_train (primal-style
      predict, but without sklearn's float64 copy)
    - If n_targets >= n_samples: keeps X_train and predicts via kernel projection
      (avoids materializing the large coef_ matrix)

    Falls back to sklearn Ridge when n_samples >= n_features.
    Mathematically identical to sklearn Ridge with fit_intercept=True.
    """

    def __init__(self, alpha: float = 1.0, chunk_size: int = 5000):
        self.alpha = alpha
        self.chunk_size = chunk_size

    def fit(self, X, Y) -> None:
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        if n_samples >= n_features:
            self._use_dual = False
            self._primal = Ridge(alpha=self.alpha)
            self._primal.fit(X, Y)
            return

        self._use_dual = True
        self._X_mean = X.mean(axis=0)
        self._Y_mean = Y.mean(axis=0)
        X_c = X - self._X_mean
        Y_c = Y - self._Y_mean

        # Compute kernel and solve in float64 for numerical stability
        K = np.float64(X_c @ X_c.T)
        K[np.diag_indices_from(K)] += self.alpha
        K_inv = np.float32(np.linalg.solve(K, np.eye(n_samples)))

        if n_targets < n_samples:
            # coef_ is smaller than X_train — compute it, free X
            dual_coef = K_inv @ Y_c
            self._coef = X_c.T @ dual_coef
            self._use_coef = True
        else:
            # X_train is smaller than coef_ — keep it for projection
            self._X_train_centered = X_c
            self._Y_train_centered = Y_c
            self._K_inv = K_inv
            self._use_coef = False

    def predict(self, X) -> np.ndarray:
        if not self._use_dual:
            return self._primal.predict(X)

        X = np.asarray(X, dtype=np.float32)
        X_test_c = X - self._X_mean

        if self._use_coef:
            return X_test_c @ self._coef + self._Y_mean

        proj = X_test_c @ self._X_train_centered.T @ self._K_inv

        n_test = X.shape[0]
        n_targets = self._Y_train_centered.shape[1]
        predictions = np.empty((n_test, n_targets), dtype=np.float32)
        for i in range(0, n_targets, self.chunk_size):
            end = min(i + self.chunk_size, n_targets)
            predictions[:, i:end] = proj @ self._Y_train_centered[:, i:end] + self._Y_mean[i:end]
        return predictions


class DualRidgeCVRegression:
    """RidgeCV with dual form for memory efficiency.

    When n_samples < n_features and no custom scoring/cv is requested,
    selects alpha via LOO cross-validation in kernel space using the
    eigendecomposition of K = X @ X.T, then predicts via dual-form
    projection. Never materializes the (n_features, n_targets) coef_ matrix.

    When custom scoring or cv is requested, falls back to sklearn RidgeCV
    for alpha selection (preserving all sklearn behavior), then uses
    DualRidgeRegression for prediction.

    Falls back to sklearn RidgeCV entirely when n_samples >= n_features.

    Exposes ``alpha_`` after fit (selected regularization strength).
    """

    def __init__(self, alphas=None, chunk_size: int = 5000, **ridgecv_kwargs):
        self.alphas = alphas
        self.chunk_size = chunk_size
        self._ridgecv_kwargs = ridgecv_kwargs
        self.alpha_ = None

    def _can_use_dual_loo(self) -> bool:
        """Check if we can do alpha selection in kernel space.

        Dual LOO is only valid when sklearn would use its efficient LOO path:
        no custom scoring function, no explicit cv folds, no per-target alpha.
        """
        if self._ridgecv_kwargs.get('scoring') is not None:
            return False
        if self._ridgecv_kwargs.get('cv') is not None:
            return False
        if self._ridgecv_kwargs.get('alpha_per_target', False):
            return False
        return True

    def fit(self, X, Y) -> None:
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        n_samples, n_features = X.shape

        if n_samples >= n_features:
            self._use_dual = False
            kwargs = dict(self._ridgecv_kwargs)
            if self.alphas is not None:
                kwargs['alphas'] = self.alphas
            self._primal = RidgeCV(**kwargs)
            self._primal.fit(X, Y)
            self.alpha_ = self._primal.alpha_
            return

        self._use_dual = True

        if self._can_use_dual_loo():
            self._fit_dual_loo(X, Y, n_samples)
        else:
            self._fit_sklearn_then_dual(X, Y)

    def _fit_dual_loo(self, X, Y, n_samples) -> None:
        """Select alpha via LOO in kernel space. No coef_ materialized.

        Replicates sklearn's _RidgeGCV eigen decomposition approach:
        center X, add intercept to kernel via outer product, eigendecompose,
        zero regularization on the intercept eigenvector, then evaluate LOO
        for each alpha candidate.

        Data stored in float32 to halve memory. Kernel eigendecomposition and
        LOO scoring done in float64 for numerical precision.
        """
        # Center
        self._X_mean = X.mean(axis=0)
        self._Y_mean = Y.mean(axis=0)
        X_c = X - self._X_mean
        Y_c = Y - self._Y_mean

        # Kernel with intercept in float64 for eigendecomposition precision
        K = np.float64(X_c @ X_c.T)
        K += 1.0  # equivalent to np.ones((n,n)) but avoids allocation

        eigenvalues, Q = np.linalg.eigh(K)
        QT_y = Q.T @ np.float64(Y)  # project UN-centered Y in float64

        # Find the intercept eigenvector (most aligned with ones vector)
        normalized_sw = np.ones(n_samples) / np.sqrt(n_samples)
        intercept_dim = np.argmax(np.abs(Q.T @ normalized_sw))

        # Evaluate LOO for each alpha (all float64 — small matrices)
        alphas = self.alphas if self.alphas is not None else [0.1, 1.0, 10.0]
        best_alpha = alphas[0]
        best_score = -np.inf

        Q_sq = Q ** 2

        for alpha in alphas:
            w = 1.0 / (eigenvalues + alpha)
            w[intercept_dim] = 0  # no regularization on intercept

            c = Q @ (w[:, None] * QT_y)
            G_inv_diag = Q_sq @ w
            G_inv_diag = np.maximum(G_inv_diag, 1e-12)

            loo_errors = c / G_inv_diag[:, None]
            score = -np.mean(loo_errors ** 2)  # negative MSE (higher is better)

            if score > best_score:
                best_score = score
                best_alpha = alpha

        self.alpha_ = best_alpha

        # Compute K_inv in float64, store as float32
        K_pred = np.float64(X_c @ X_c.T)
        K_pred[np.diag_indices_from(K_pred)] += self.alpha_
        K_inv = np.float32(np.linalg.solve(K_pred, np.eye(n_samples)))

        # Adaptive storage: keep whichever is smaller after fit
        n_targets = Y_c.shape[1]
        if n_targets < n_samples:
            # coef_ is smaller than X_train — compute it, free X
            dual_coef = K_inv @ Y_c
            self._coef = X_c.T @ dual_coef
            self._use_coef = True
        else:
            # X_train is smaller than coef_ — keep it for projection
            self._X_train_centered = X_c
            self._Y_train_centered = Y_c
            self._K_inv = K_inv
            self._use_coef = False

    def _fit_sklearn_then_dual(self, X, Y) -> None:
        """Fallback: sklearn RidgeCV for alpha, DualRidge for prediction.

        Used when custom scoring/cv/alpha_per_target prevents dual LOO.
        """
        kwargs = dict(self._ridgecv_kwargs)
        if self.alphas is not None:
            kwargs['alphas'] = self.alphas
        rcv = RidgeCV(**kwargs)
        rcv.fit(X, Y)
        self.alpha_ = rcv.alpha_
        del rcv

        self._dual = DualRidgeRegression(
            alpha=float(self.alpha_), chunk_size=self.chunk_size
        )
        self._dual.fit(X, Y)

    def predict(self, X) -> np.ndarray:
        if not self._use_dual:
            return self._primal.predict(X)

        if hasattr(self, '_dual'):
            return self._dual.predict(X)

        X = np.asarray(X, dtype=np.float32)
        X_test_c = X - self._X_mean

        if self._use_coef:
            return X_test_c @ self._coef + self._Y_mean

        proj = X_test_c @ self._X_train_centered.T @ self._K_inv

        n_test = X.shape[0]
        n_targets = self._Y_train_centered.shape[1]
        predictions = np.empty((n_test, n_targets), dtype=np.float32)
        for i in range(0, n_targets, self.chunk_size):
            end = min(i + self.chunk_size, n_targets)
            predictions[:, i:end] = (
                proj @ self._Y_train_centered[:, i:end] + self._Y_mean[i:end]
            )
        return predictions


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


def dual_ridge_regression(regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(alpha=1)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = DualRidgeRegression(**regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def dual_ridge_cv_regression(regression_kwargs=None, xarray_kwargs=None, alphas=ALPHA_LIST):
    regression_defaults = dict(alphas=alphas)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression_kwargs.pop('alpha', None)

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
