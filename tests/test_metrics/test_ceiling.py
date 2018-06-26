from mkgu.metrics import Score
from mkgu.metrics.ceiling import NoCeiling


class TestNoCeiling:
    def test(self):
        ceiling = NoCeiling()
        ceiling = ceiling(None)
        assert isinstance(ceiling, Score)
