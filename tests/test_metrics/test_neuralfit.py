import numpy as np
import xarray
from pytest import approx

import mkgu
from mkgu.metrics.neural_fit import NeuralFitMetric, PCANeuroidCharacterization
from tests.test_metrics import load_hvm


class TestNeuralFit(object):
    def test_hvm_self(self):
        hvm = load_hvm()
        neural_fit_metric = NeuralFitMetric()
        score = neural_fit_metric(hvm, hvm)
        assert score.center == approx(0.78, rel=0.005)


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
