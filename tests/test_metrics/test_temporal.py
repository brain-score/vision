import numpy as np
import scipy.stats

from brainio.assemblies import NeuroidAssembly
from brainscore.metrics.regression import pls_regression, pearsonr_correlation
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages, \
    TemporalCorrelationAcrossTime
from brainscore.metrics.xarray_utils import XarrayCorrelation


class TestTemporalRegressionAcrossTime:
    def test_small(self):
        values = (np.arange(30 * 25 * 5) + np.random.standard_normal(30 * 25 * 5)).reshape((30, 25, 5))
        assembly = NeuroidAssembly(values,
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', ['some_region'] * 25),
                                           'time_bin_start': ('time_bin', list(range(5))),
                                           'time_bin_end': ('time_bin', list(range(1, 6))),
                                           },
                                   dims=['presentation', 'neuroid', 'time_bin'])
        regression = TemporalRegressionAcrossTime(pls_regression())
        regression.fit(source=assembly, target=assembly)
        prediction = regression.predict(source=assembly)
        assert all(prediction['stimulus_id'] == assembly['stimulus_id'])
        assert all(prediction['neuroid_id'] == assembly['neuroid_id'])
        assert all(prediction['time_bin'] == assembly['time_bin'])


class TestTemporalCorrelation:
    def test_across_images(self):
        values = (np.arange(30 * 25 * 5) + np.random.standard_normal(30 * 25 * 5)).reshape((30, 25, 5))
        assembly = NeuroidAssembly(values,
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', ['some_region'] * 25),
                                           'time_bin_start': ('time_bin', list(range(5))),
                                           'time_bin_end': ('time_bin', list(range(1, 6))),
                                           },
                                   dims=['presentation', 'neuroid', 'time_bin'])
        correlation = TemporalCorrelationAcrossImages(pearsonr_correlation())
        score = correlation(assembly, assembly)
        np.testing.assert_array_equal(score.dims, ['neuroid'])
        np.testing.assert_array_equal(score['neuroid_id'].values, list(range(25)))
        np.testing.assert_array_almost_equal(score.values, [1.] * 25)
        assert set(score.raw.dims) == {'neuroid', 'time_bin'}

    def test_across_time(self):
        values = (np.arange(30 * 25 * 5) + np.random.standard_normal(30 * 25 * 5)).reshape((30, 25, 5))
        assembly = NeuroidAssembly(values,
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', ['some_region'] * 25),
                                           'time_bin_start': ('time_bin', list(range(5))),
                                           'time_bin_end': ('time_bin', list(range(1, 6))),
                                           },
                                   dims=['presentation', 'neuroid', 'time_bin'])
        correlation = XarrayCorrelation(scipy.stats.pearsonr, correlation_coord='time_bin')
        correlation = TemporalCorrelationAcrossTime(correlation)
        score = correlation(assembly, assembly)
        np.testing.assert_array_equal(score.dims, ['neuroid'])
        np.testing.assert_array_equal(score['neuroid_id'].values, list(range(25)))
        np.testing.assert_array_almost_equal(score.values, [1.] * 25)
        assert set(score.raw.dims) == {'neuroid', 'presentation'}
