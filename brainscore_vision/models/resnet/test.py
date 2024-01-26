import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@pytest.mark.travis_slow
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("resnet-18", "MajajHong2015public.IT-pls", approx(0.549, abs=0.001)),
    ("resnet-34", "MajajHong2015public.IT-pls", approx(0.543, abs=0.001)),
    ("resnet-50", "MajajHong2015public.IT-pls", approx(0.537, abs=0.001)),
    ("resnet-101", "MajajHong2015public.IT-pls", approx(0.528, abs=0.001)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    seed_everything(42)
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
