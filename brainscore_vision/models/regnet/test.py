import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@pytest.mark.travis_slow
@pytest.mark.memory_intense
def test_score():
    seed_everything(42)
    actual_score = score(model_identifier="regnet_y_400mf", benchmark_identifier="MajajHong2015public.IT-pls",
                         conda_active=True)
    assert actual_score == approx(0.532, abs=0.0005)
