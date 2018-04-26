import numpy as np
import xarray

import mkgu
from mkgu.metrics.neural_fit import NeuralFitMetric, PCANeuroidCharacterization
from tests.test_metrics import load_hvm


class TestPCA(object):
    def test_noop(self):
        hvm = load_hvm()
        pca = PCANeuroidCharacterization(max_components=1000)
        hvm_ = pca(hvm)
        xarray.testing.assert_equal(hvm, hvm_)

    def test_100(self):
        hvm = load_hvm()
        pca = PCANeuroidCharacterization(max_components=100)
        hvm_ = pca(hvm)
        assert isinstance(hvm_, mkgu.assemblies.NeuroidAssembly)
        np.testing.assert_array_equal([hvm.shape[0], 100], hvm_.shape)


class TestNeuralFit(object):
    def test_nopca(self):
        hvm = load_hvm()
        neural_fit_metric = NeuralFitMetric(pca_components=None)
        score = neural_fit_metric(hvm, hvm)
        assert 0.75 < score < 0.8

    def test_pca100(self):
        hvm = load_hvm()
        neural_fit_metric = NeuralFitMetric(pca_components=100)
        score = neural_fit_metric(hvm, hvm)
        assert 0.10 < score < 0.15
