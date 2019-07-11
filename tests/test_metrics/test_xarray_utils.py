import numpy as np
import scipy.stats
from pytest import approx
from sklearn.linear_model import LinearRegression

from brainio_base.assemblies import NeuroidAssembly
from brainscore.metrics.xarray_utils import XarrayRegression, XarrayCorrelation


class TestXarrayRegression:
    def test_fitpredict_alignment(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10),
                                         coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10)},
                                         dims=['presentation', 'neuroid'])
        target = jumbled_source.sortby(['image_id', 'neuroid_id'])
        regression = XarrayRegression(LinearRegression())
        regression.fit(jumbled_source, target)
        prediction = regression.predict(jumbled_source)
        # do not test for alignment of metadata - it is only important that the data is well-aligned with the metadata.
        np.testing.assert_array_almost_equal(prediction.sortby(['image_id', 'neuroid_id']).values,
                                             target.sortby(['image_id', 'neuroid_id']).values)

    def test_neuroid_single_coord(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10),
                                         coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid_id', list(reversed(range(10))))},
                                         dims=['presentation', 'neuroid_id']).stack(neuroid=['neuroid_id'])
        target = jumbled_source.sortby(['image_id', 'neuroid_id'])
        regression = XarrayRegression(LinearRegression())
        regression.fit(jumbled_source, target)
        prediction = regression.predict(jumbled_source)
        assert set(prediction.dims) == {'presentation', 'neuroid'}
        assert len(prediction['neuroid_id']) == 10


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
        assert all(score == approx(1))

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
        assert all(score == approx(1))

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
