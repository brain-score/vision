import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@pytest.mark.travis_slow
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("densenet121", "MajajHong2015public.IT-pls", approx(0.548, abs=0.001)),
    ("densenet161", "MajajHong2015public.IT-pls", approx(0.561, abs=0.001)),
    ("densenet169", "MajajHong2015public.IT-pls", approx(0.556, abs=0.001)),
    ("densenet201", "MajajHong2015public.IT-pls", approx(0.561, abs=0.001)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True,
    )
    assert actual_score == expected_score
