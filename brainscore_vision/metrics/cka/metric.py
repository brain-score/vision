import math

import numpy as np
from brainscore_core import Metric

from brainio.assemblies import DataAssembly
from brainscore_vision.metrics import Score
from brainscore_vision.metric_helpers.transformations import TestOnlyCrossValidation
from brainscore_vision.metric_helpers import Defaults as XarrayDefaults


class CKACrossValidated(Metric):
    """
    Computes a cross-validated similarity index for the similarity between two assemblies
    with centered kernel alignment (CKA).

    Kornblith et al., 2019 http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf
    """

    def __init__(self, comparison_coord=XarrayDefaults.stimulus_coord, crossvalidation_kwargs=None):
        self._metric = CKAMetric(comparison_coord=comparison_coord)
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

    def __init__(self, comparison_coord=XarrayDefaults.stimulus_coord):
        self._comparison_coord = comparison_coord

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
        similarity = linear_CKA(assembly1, assembly2)
        return Score(similarity)


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)
    # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the sme with one time centering
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
