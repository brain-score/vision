import logging
import sys

import numpy as np
import xarray
from pytest import approx

import mkgu
from mkgu.metrics import CartesianProduct, CrossValidation
from mkgu.metrics.neural_fit import NeuralFit, PCA
from tests.test_metrics import load_hvm

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class TestNeuralFit(object):
    def test_hvm_pls_regions(self):
        hvm = load_hvm()
        neural_fit_metric = NeuralFit(regression='pls-25')
        score = neural_fit_metric(hvm, hvm)
        expected_scores = {'V4': 0.795, 'IT': 0.826}
        for region in ['V4', 'IT']:
            assert score.center.sel(region_left=region, region_right=region) \
                   == approx(expected_scores[region], abs=0.01), \
                "region {} score does not match".format(region)

    def test_hvm_linear_subregions(self):
        hvm = load_hvm()
        transformations = [CartesianProduct(dividing_coord_names=('subregion',)), CrossValidation()]
        neural_fit_metric = NeuralFit(regression='linear', transformations=transformations)
        score = neural_fit_metric(hvm, hvm)
        for subregion in ['V4', 'pIT', 'cIT', 'aIT']:
            assert score.center.sel(subregion_left=subregion, subregion_right=subregion) == approx(1, rel=0.005), \
                "subregion {} score does not match".format(subregion)


class TestPCA(object):
    def test_noop(self):
        hvm = load_hvm().sel(region='IT')
        pca = PCA(max_components=1000)
        hvm_ = pca(hvm)
        xarray.testing.assert_equal(hvm, hvm_)

    def test_100(self):
        hvm = load_hvm().sel(region='IT')
        pca = PCA(max_components=100)
        hvm_ = pca(hvm)
        assert isinstance(hvm_, mkgu.assemblies.NeuroidAssembly)
        np.testing.assert_array_equal([hvm.shape[0], 100], hvm_.shape)
