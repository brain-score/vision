import numpy as np
from brainio_base.assemblies import NeuroidAssembly
from sklearn.linear_model import LinearRegression

from model_tools.xarray_utils import XarrayRegression


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
