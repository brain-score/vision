from pytest import approx
from brainscore_vision.metrics.value_delta import ValueDelta


class TestValueDelta:

    def test_perfect_score(self):
        metric = ValueDelta()
        value_delta = metric(1.25, 1.25)
        assert value_delta == 1

    def test_middle_score(self):
        metric = ValueDelta()
        value_delta = metric(1.50, 1.00)
        assert value_delta == approx(0.6065, abs=.0005)

    def test_middle_score_reversed(self):
        metric = ValueDelta()
        value_delta = metric(1.00, 1.50)
        assert value_delta == approx(0.6065, abs=.0005)

    def test_bad_score(self):
        metric = ValueDelta()
        value_delta = metric(1.00, -1.00)
        assert value_delta == approx(0.1353, abs=.0005)

    def test_bad_score_reversed(self):
        metric = ValueDelta()
        value_delta = metric(-1.00, 1.00)
        assert value_delta == approx(0.1353, abs=.0005)

    def test_too_high_score(self):
        metric = ValueDelta()
        value_delta = metric(-5.00, 5.00)
        assert value_delta == approx(0.0, abs=.0005)

    def test_too_high_score_reversed(self):
        metric = ValueDelta()
        value_delta = metric(5.00, -5.00)
        assert value_delta == approx(0.0, abs=.0005)
