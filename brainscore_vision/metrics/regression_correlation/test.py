import numpy as np
import pytest
from pytest import approx
from sklearn.datasets import make_regression

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_vision import load_metric
from sklearn.linear_model import RidgeCV

from .metric import pls_regression, linear_regression, ridge_regression, ridge_cv_regression, \
    DualRidgeCVRegression, ALPHA_LIST


class TestCrossRegressedCorrelation:
    def test_small(self):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', ['some_region'] * 25)},
                                   dims=['presentation', 'neuroid'])
        metric = load_metric('pls')
        score = metric(source=assembly, target=assembly)
        assert score == approx(1, abs=.00001)


class TestRegression:
    @pytest.mark.parametrize('regression_ctr', [pls_regression, linear_regression, ridge_regression, ridge_cv_regression])
    def test_small(self, regression_ctr):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', [None] * 25)},
                                   dims=['presentation', 'neuroid'])
        regression = regression_ctr()
        regression.fit(source=assembly, target=assembly)
        prediction = regression.predict(source=assembly)
        assert all(prediction['stimulus_id'] == assembly['stimulus_id'])
        assert all(prediction['neuroid_id'] == assembly['neuroid_id'])

class TestTrainTestSplitCorrelation:
    @pytest.mark.parametrize('metric_name', ['pls_split', 'ridge_split', 'linear_predictivity_split', 'neuron_to_neuron_split', 'ridgecv_split'])
    def test_small(self, metric_name):
        train_assembly = NeuroidAssembly(
            (np.arange(100 * 25) + np.random.standard_normal(100 * 25)).reshape((100, 25)),
            coords={'stimulus_id': ('presentation', np.arange(100)),
                    'object_name': ('presentation', ['a', 'b', 'c', 'd', 'e'] * 20),
                    'neuroid_id': ('neuroid', np.arange(25)),
                    'region': ('neuroid', ['some_region'] * 25)},
            dims=['presentation', 'neuroid'])
        test_assembly = NeuroidAssembly(
            (np.arange(20 * 25) + np.random.standard_normal(20 * 25)).reshape((20, 25)),
            coords={'stimulus_id': ('presentation', np.arange(20)),
                    'object_name': ('presentation', ['f', 'g', 'h', 'i'] * 5),
                    'neuroid_id': ('neuroid', np.arange(25)),
                    'region': ('neuroid', ['some_region'] * 25)},
            dims=['presentation', 'neuroid'])
        metric = load_metric(metric_name)
        score = metric(source_train=train_assembly, source_test=test_assembly,
                      target_train=train_assembly, target_test=test_assembly)
        assert len(score.raw) == 25
        assert score == approx(1, abs=.001)
        assert all(score.raw['neuroid_id'] == test_assembly['neuroid_id'])


class TestDualRidgeCV:
    """The dual form exists to avoid the (n_features, n_targets) coef_ matrix.
    Alpha selection must not rebuild it, or peak memory is unchanged.
    """

    @staticmethod
    def _data(seed, n=60, p=800, q=300, noise=1.0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, p)
        return X, X @ rng.randn(p, q) / np.sqrt(p) + noise * rng.randn(n, q)

    @pytest.mark.parametrize('seed', range(3))
    @pytest.mark.parametrize('noise', [0.05, 1.0, 20.0])
    def test_matches_sklearn(self, seed, noise):
        X, Y = self._data(seed, noise=noise)
        reference = RidgeCV(alphas=ALPHA_LIST)
        reference.fit(X, Y)
        regression = DualRidgeCVRegression(alphas=ALPHA_LIST)
        regression.fit(X, Y)
        assert regression.alpha_ == approx(reference.alpha_, rel=1e-9)
        assert regression.predict(X) == approx(reference.predict(X), abs=1e-6)

    def test_alpha_selection_design_is_sample_sized(self, monkeypatch):
        X, Y = self._data(0)
        designs = []
        original = RidgeCV.fit
        monkeypatch.setattr(RidgeCV, 'fit',
                            lambda self, design, y, **kwargs: designs.append(design.shape)
                            or original(self, design, y, **kwargs))
        DualRidgeCVRegression(alphas=ALPHA_LIST).fit(X, Y)
        n_samples, n_features = X.shape
        assert designs == [(n_samples, n_samples)], f'alpha selection saw {designs}, not the gram surrogate'
        assert n_features > n_samples  # otherwise the dual branch was never taken
