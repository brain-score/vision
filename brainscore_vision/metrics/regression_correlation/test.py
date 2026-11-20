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


class TestPLSFewerFeaturesThanComponents:
    """ PLS defaults to 25 components, which used to raise for sources with fewer than 25 features (#335) """

    @staticmethod
    def _assembly(values):
        n_stimuli, n_neuroids = values.shape
        return NeuroidAssembly(values, coords={'stimulus_id': ('presentation', np.arange(n_stimuli)),
                                               'object_name': ('presentation', ['a', 'b', 'c'] * (n_stimuli // 3)),
                                               'neuroid_id': ('neuroid', np.arange(n_neuroids)),
                                               'region': ('neuroid', ['some_region'] * n_neuroids)},
                               dims=['presentation', 'neuroid'])

    @pytest.mark.parametrize('n_features', [1, 10, 24])
    def test_matches_least_squares(self, n_features):
        # with as many components as source features, PLS fits the least-squares model
        rng = np.random.RandomState(0)
        source = self._assembly(rng.standard_normal((60, n_features)))
        target = self._assembly(source.values @ rng.standard_normal((n_features, 5)) + rng.standard_normal((60, 5)))
        pls, least_squares = pls_regression(), linear_regression()
        pls.fit(source=source, target=target)
        least_squares.fit(source=source, target=target)
        test_source = self._assembly(rng.standard_normal((30, n_features)))
        assert pls.predict(test_source).values == approx(least_squares.predict(test_source).values, abs=1e-8)

    def test_rank_deficient_matches_least_squares(self):
        # a dead unit and a duplicated unit: components beyond the source rank would only fit round-off
        rng = np.random.RandomState(0)
        def with_degenerate_units(values):
            return self._assembly(np.hstack([values, np.zeros((len(values), 1)), values[:, :1]]))
        source = with_degenerate_units(rng.standard_normal((60, 8)))
        target = self._assembly(source.values @ rng.standard_normal((10, 5)) + rng.standard_normal((60, 5)))
        pls, least_squares = pls_regression(), linear_regression()
        pls.fit(source=source, target=target)
        least_squares.fit(source=source, target=target)
        test_source = with_degenerate_units(rng.standard_normal((30, 8)))
        assert pls.predict(test_source).values == approx(least_squares.predict(test_source).values, abs=1e-8)

    @pytest.mark.parametrize('metric_name, n_source, n_target', [('pls', 10, 30), ('reverse_pls', 30, 10)])
    def test_metric(self, metric_name, n_source, n_target):
        rng = np.random.RandomState(0)
        latent = rng.standard_normal((90, 5))
        source = self._assembly(latent @ rng.standard_normal((5, n_source)) + .1 * rng.standard_normal((90, n_source)))
        target = self._assembly(latent @ rng.standard_normal((5, n_target)) + .1 * rng.standard_normal((90, n_target)))
        metric = load_metric(metric_name, crossvalidation_kwargs=dict(stratification_coord=None))
        score = metric(source, target)
        assert score > .9


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
