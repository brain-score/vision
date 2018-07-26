import os

import xarray as xr
from pytest import approx

from brainscore.metrics.behavior import I2n


class TestI2N(object):
    def test_objectome(self):
        objectome = xr.open_dataarray(os.path.join(os.path.dirname(__file__), 'monkobjectome_behavior.nc'))
        i2n = I2n()
        score = i2n(train_source=objectome, train_target=objectome, test_source=objectome, test_target=objectome)
        score = i2n.aggregate(score)
        expected_score = 0.826
        assert score == approx(expected_score, abs=0.01)
