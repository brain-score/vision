import numpy as np
from mkgu.metrics import Score
from mkgu.metrics.ceiling import NoCeiling, SplitNoCeiling


class TestNoCeiling:
    def test(self):
        ceiling = NoCeiling()
        ceiling = ceiling(None)
        assert isinstance(ceiling, Score)


class TestSplitNoCeiling:
    def test(self):
        ceiling = SplitNoCeiling()
        ceiling = ceiling(None)
        assert isinstance(ceiling, Score)
        np.testing.assert_array_equal(ceiling.values, [1] * 10)
        assert ceiling.aggregation.sel(aggregation='center') == 1
        assert ceiling.aggregation.sel(aggregation='error') == 0
