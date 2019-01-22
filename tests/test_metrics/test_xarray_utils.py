import numpy as np
import scipy.stats

from brainio_base.assemblies import NeuroidAssembly
from brainscore.metrics.xarray_utils import XarrayCorrelation


class TestXarrayCorrelation:
    def test_dimensions(self):
        prediction = NeuroidAssembly(np.random.rand(500, 10),
                                     coords={'image_id': ('presentation', list(range(500))),
                                             'image_meta': ('presentation', [0] * 500),
                                             'neuroid_id': ('neuroid', list(range(10))),
                                             'neuroid_meta': ('neuroid', [0] * 10)},
                                     dims=['presentation', 'neuroid'])
        correlation = XarrayCorrelation(lambda a, b: (1, 0))
        score = correlation(prediction, prediction)
        np.testing.assert_array_equal(score.dims, ['neuroid'])
        np.testing.assert_array_equal(score.shape, [10])

    def test_correlation(self):
        prediction = NeuroidAssembly(np.random.rand(500, 10),
                                     coords={'image_id': ('presentation', list(range(500))),
                                             'image_meta': ('presentation', [0] * 500),
                                             'neuroid_id': ('neuroid', list(range(10))),
                                             'neuroid_meta': ('neuroid', [0] * 10)},
                                     dims=['presentation', 'neuroid'])
        correlation = XarrayCorrelation(lambda a, b: (1, 0))
        score = correlation(prediction, prediction)
        assert all(score == 1)

    def test_alignment(self):
        jumbled_prediction = NeuroidAssembly(np.random.rand(500, 10),
                                             coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                     'image_meta': ('presentation', [0] * 500),
                                                     'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                     'neuroid_meta': ('neuroid', [0] * 10)},
                                             dims=['presentation', 'neuroid'])
        prediction = jumbled_prediction.sortby(['image_id', 'neuroid_id'])
        correlation = XarrayCorrelation(scipy.stats.pearsonr)
        score = correlation(jumbled_prediction, prediction)
        assert all(score == 1)

    def test_neuroid_single_coord(self):
        prediction = NeuroidAssembly(np.random.rand(500, 10),
                                     coords={'image_id': ('presentation', list(range(500))),
                                             'image_meta': ('presentation', [0] * 500),
                                             'neuroid_id': ('neuroid_id', list(range(10)))},
                                     dims=['presentation', 'neuroid_id']).stack(neuroid=['neuroid_id'])
        correlation = XarrayCorrelation(lambda a, b: (1, 0))
        score = correlation(prediction, prediction)
        np.testing.assert_array_equal(score.dims, ['neuroid'])
        assert len(score['neuroid']) == 10
