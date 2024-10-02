import numpy as np
import scipy.stats
from pytest import approx
from sklearn.linear_model import LinearRegression

from brainio.assemblies import NeuroidAssembly
from brainscore_vision.metric_helpers.xarray_utils import XarrayRegression, XarrayCorrelation
from brainscore_vision.metric_helpers.temporal import PerTime, SpanTime, PerTimeRegression, SpanTimeRegression


class TestMetricHelpers:
    def test_pertime_ops(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10, 20),
                                         coords={'stimulus_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10),
                                                 'time_bin_start': ('time_bin', np.arange(0, 400, 20)),
                                                 'time_bin_end': ('time_bin', np.arange(20, 420, 20))},
                                         dims=['presentation', 'neuroid', 'time_bin'])
        mean_neuroid = lambda arr: arr.mean('neuroid')
        pertime_mean_neuroid = PerTime(mean_neuroid)
        output = pertime_mean_neuroid(jumbled_source)
        output = output.transpose('presentation', 'time_bin')
        target = jumbled_source.transpose('presentation', 'time_bin', 'neuroid').mean('neuroid')
        assert (output == approx(target)).all().item()

    def test_spantime_ops(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10, 20),
                                         coords={'stimulus_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10),
                                                 'time_bin_start': ('time_bin', np.arange(0, 400, 20)),
                                                 'time_bin_end': ('time_bin', np.arange(20, 420, 20))},
                                         dims=['presentation', 'neuroid', 'time_bin'])
        mean_presentation = lambda arr: arr.mean("presentation")
        spantime_mean_presentation = SpanTime(mean_presentation)
        output = spantime_mean_presentation(jumbled_source)
        output = output.transpose('neuroid')
        target = jumbled_source.transpose('presentation', 'time_bin', 'neuroid').mean('presentation').mean('time_bin')
        assert (output == approx(target)).all().item()

    def test_pertime_regression(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10, 20),
                                         coords={'stimulus_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10),
                                                 'time_bin_start': ('time_bin', np.arange(0, 400, 20)),
                                                 'time_bin_end': ('time_bin', np.arange(20, 420, 20))},
                                         dims=['presentation', 'neuroid', 'time_bin'])
        target = jumbled_source.sortby(['stimulus_id', 'neuroid_id'])
        pertime_regression = PerTimeRegression(XarrayRegression(LinearRegression()))
        pertime_regression.fit(jumbled_source, target)
        prediction = pertime_regression.predict(jumbled_source)
        prediction = prediction.transpose(*target.dims)
        # do not test for alignment of metadata - it is only important that the data is well-aligned with the metadata.
        np.testing.assert_array_almost_equal(prediction.sortby(['stimulus_id', 'neuroid_id', 'time_bin']).values,
                                             target.sortby(['stimulus_id', 'neuroid_id', 'time_bin']).values)


    def test_spantime_regression(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10, 20),
                                         coords={'stimulus_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10),
                                                 'time_bin_start': ('time_bin', np.arange(0, 400, 20)),
                                                 'time_bin_end': ('time_bin', np.arange(20, 420, 20))},
                                         dims=['presentation', 'neuroid', 'time_bin'])
        target = jumbled_source.sortby(['stimulus_id', 'neuroid_id'])
        spantime_regression = SpanTimeRegression(XarrayRegression(LinearRegression()))
        spantime_regression.fit(jumbled_source, target)
        prediction = spantime_regression.predict(jumbled_source)
        prediction = prediction.transpose(*target.dims)
        # do not test for alignment of metadata - it is only important that the data is well-aligned with the metadata.
        np.testing.assert_array_almost_equal(prediction.sortby(['stimulus_id', 'neuroid_id', 'time_bin']).values,
                                             target.sortby(['stimulus_id', 'neuroid_id', 'time_bin']).values)

