import numpy as np
import xarray
from pytest import approx

import brainscore
from brainscore.benchmarks import SplitBenchmark
from brainscore.metrics.ceiling import SplitNoCeiling
from brainscore.metrics.neural_fit import NeuralFit, PCA, Pls25Fit
from tests.test_metrics import load_hvm


class TestNeuralFit(object):
    def test_hvm_pls_IT(self):
        hvm = load_hvm()
        hvm = hvm.sel(region='IT')
        neural_fit = Pls25Fit()
        score = neural_fit(train_source=hvm, train_target=hvm, test_source=hvm, test_target=hvm)
        score = neural_fit.aggregate(score)
        expected_score = 0.826
        assert score == approx(expected_score, abs=0.01)

    def test_hvm_pls_regions(self):
        hvm = load_hvm()
        neural_fit = Pls25Fit()
        benchmark = SplitBenchmark(metric=neural_fit, target_assembly=hvm, ceiling=SplitNoCeiling(),
                                   target_splits=['region'])
        score = benchmark(hvm, source_splits=['region'])
        expected_scores = {'V4': 0.795, 'IT': 0.826}
        for region in ['V4', 'IT']:
            assert score.aggregation.sel(aggregation='center', region_source=region, region_target=region) \
                   == approx(expected_scores[region], abs=0.01), \
                "region {} score does not match".format(region)

    def test_hvm_linear_subregions(self):
        hvm = load_hvm()
        neural_fit = NeuralFit()
        benchmark = SplitBenchmark(metric=neural_fit, target_assembly=hvm, ceiling=SplitNoCeiling(),
                                   target_splits=['subregion'])
        score = benchmark(hvm, source_splits=['subregion'])
        for subregion in ['V4', 'pIT', 'cIT', 'aIT']:
            assert score.aggregation.sel(aggregation='center', subregion_source=subregion, subregion_target=subregion) \
                   == approx(1, rel=0.005), \
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
        assert isinstance(hvm_, brainscore.assemblies.NeuroidAssembly)
        np.testing.assert_array_equal([hvm.shape[0], 100], hvm_.shape)
