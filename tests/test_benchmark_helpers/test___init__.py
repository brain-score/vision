import numpy as np
import pytest

from brainscore_core import Score
from brainscore_vision.benchmark_helpers import bound_score


class TestBoundScore:
    @pytest.mark.parametrize('value', [
        .42,
        0,
        1,
        .99999,
        0.00000001,
    ])
    def test_nothing_to_do(self, value):
        score = Score(value)
        bound_score(score)
        assert score == value

    def test_nothing_to_do_nan(self):
        score = Score(np.nan)
        bound_score(score)
        assert np.isnan(score)

    @pytest.mark.parametrize('value', [
        -0.1,
        -0.0000001,
        -1,
        -9999,
    ])
    def test_negative_to_0(self, value):
        score = Score(value)
        bound_score(score)
        assert score == Score(0)

    @pytest.mark.parametrize('value', [
        1.1,
        1.0000001,
        2,
        9999,
    ])
    def test_greater_1_to_1(self, value):
        score = Score(value)
        bound_score(score)
        assert score == Score(1)

    @pytest.mark.parametrize('value', [
        None,
        [0.3, 0.2],
        'abc',
    ])
    def test_expected_fails(self, value):
        score = Score(value)
        with pytest.raises((TypeError, ValueError)):
            bound_score(score)

    def test_retains_attrs(self):
        score = Score(.42)
        score.attrs['ceiling'] = score
        bound_score(score)
        assert 'ceiling' in score.attrs
        assert score.attrs['ceiling'] == score
