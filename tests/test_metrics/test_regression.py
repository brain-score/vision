import numpy as np
import pytest
from pytest import approx

from brainio_base.assemblies import NeuroidAssembly
from brainscore.metrics.regression import pls_regression, linear_regression, CrossRegressedCorrelation


class TestCrossRegressedCorrelation:
    def test_small(self):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'image_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', ['some_region'] * 25)},
                                   dims=['presentation', 'neuroid'])
        metric = CrossRegressedCorrelation()
        score = metric(source=assembly, target=assembly)
        assert score.sel(aggregation='center') == approx(1, abs=.00001)


class TestRegression:
    @pytest.mark.parametrize('regression_ctr', [pls_regression, linear_regression])
    def test_small(self, regression_ctr):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'image_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', [None] * 25)},
                                   dims=['presentation', 'neuroid'])
        regression = regression_ctr()
        regression.fit(source=assembly, target=assembly)
        prediction = regression.predict(source=assembly)
        assert all(prediction['image_id'] == assembly['image_id'])
        assert all(prediction['neuroid_id'] == assembly['neuroid_id'])
