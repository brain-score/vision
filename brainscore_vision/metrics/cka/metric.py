import math

import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_core import Metric
from brainscore_vision.metric_helpers import Defaults as XarrayDefaults
from brainscore_vision.metric_helpers.transformations import TestOnlyCrossValidation, apply_aggregate
from brainscore_vision.metrics import Score


class CKACrossValidated(Metric):
    """
    Computes a cross-validated similarity index for the similarity between two assemblies
    with centered kernel alignment (CKA).

    Kornblith et al., 2019 http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf
    """

    def __init__(self, comparison_coord=XarrayDefaults.stimulus_coord, crossvalidation_kwargs=None, unbiased: bool = True):
        self._metric = CKA(comparison_coord=comparison_coord, unbiased=unbiased)
        crossvalidation_defaults = dict(test_size=.9)  # leave 10% out
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}
        self._cross_validation = TestOnlyCrossValidation(**crossvalidation_kwargs)

    def __call__(self, assembly1: DataAssembly, assembly2: DataAssembly) -> Score:
        return self._cross_validation(assembly1, assembly2, apply=self._metric)


class CKA(Metric):
    """
    Computes a similarity index for the similarity between two assemblies with centered kernel alignment (CKA).

    Kornblith et al., 2019 http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf
    """

    def __init__(self, comparison_coord=XarrayDefaults.stimulus_coord, unbiased: bool = True):
        self._comparison_coord = comparison_coord
        self._unbiased = unbiased

    def __call__(self, assembly1: DataAssembly, assembly2: DataAssembly) -> Score:
        """
        :param brainscore.assemblies.NeuroidAssembly assembly1:
        :param brainscore.assemblies.NeuroidAssembly assembly2:
        :return: brainscore.assemblies.DataAssembly
        """
        # ensure value order
        assembly1 = assembly1.sortby(self._comparison_coord)
        assembly2 = assembly2.sortby(self._comparison_coord)
        assert (assembly1[self._comparison_coord].values == assembly2[self._comparison_coord].values).all()
        # ensure dimensions order
        dims = assembly1[self._comparison_coord].dims
        np.testing.assert_array_equal(assembly2[self._comparison_coord].dims, dims)
        assembly1 = assembly1.transpose(*(list(dims) + [dim for dim in assembly1.dims if dim not in dims]))
        assembly2 = assembly2.transpose(*(list(dims) + [dim for dim in assembly2.dims if dim not in dims]))
        if self._unbiased:
            similarity = unbiased_linear_CKA(assembly1.values, assembly2.values)
        else:
            similarity = linear_CKA(assembly1, assembly2)
        return Score(similarity)


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)
    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the same with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def unbiased_linear_HSIC(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Unbiased HSIC estimator (Song et al.) for linear kernel.
        
        Supervised Feature Selection via Dependence Estimation
        https://arxiv.org/abs/0704.2668
        
        Similarity of Neural Network Representations Revisited
        https://proceedings.mlr.press/v97/kornblith19a.html
        
        Correcting Biased Centered Kernel Alignment Measures in Biological and Artificial Neural Networks
        https://arxiv.org/abs/2405.01012


        X: [n, d_x]
        Y: [n, d_y]
        Returns: scalar HSIC estimate.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}")
        n = X.shape[0]
        if n < 4:
            raise ValueError(f"Unbiased HSIC requires at least 4 samples, got n={n}")

        # Linear kernel Gram matrices
        K = X @ X.T   # [n, n]
        L = Y @ Y.T   # [n, n]

        # Zero out diagonals
        K = K.copy()
        L = L.copy()
        np.fill_diagonal(K, 0.0)
        np.fill_diagonal(L, 0.0)

        ones = np.ones((n, 1), dtype=K.dtype)

        # term1 = tr(K L) = sum_ij K_ij L_ij
        term1 = float((K * L).sum())

        # term2 = (1^T K 1)(1^T L 1)
        K1 = K @ ones   # [n, 1]
        L1 = L @ ones   # [n, 1]
        term2 = float(K1.sum() * L1.sum())

        # term3 = 1^T K L 1 = sum_i K1_i L1_i
        term3 = float((K1 * L1).sum())

        coef = 1.0 / (n * (n - 3.0))
        hsic = coef * (
            term1
            + term2 / ((n - 1.0) * (n - 2.0))
            - 2.0 * term3 / (n - 2.0)
        )
        return hsic


def unbiased_linear_CKA(X: np.ndarray, Y: np.ndarray, eps:float = 1e-12) -> float:
    """
    Unbiased CKA estimator for linear kernel.


    X: [n, d_x]
    Y: [n, d_y]
    Returns: scalar CKA estimate.
    """
    hsic_xy = unbiased_linear_HSIC(X, Y)
    hsic_xx = unbiased_linear_HSIC(X, X)
    hsic_yy = unbiased_linear_HSIC(Y, Y)
    
    prod = hsic_xx * hsic_yy
    if prod <= 0:
        return 0.0
    denom = np.sqrt(prod) + eps
    return float(hsic_xy / denom)