import numpy as np

from brainscore.assemblies import NeuroidAssembly
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
