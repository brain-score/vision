import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@pytest.mark.travis_slow
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    # Private
    pytest.param("googlenet", "MajajHong2015.IT-pls", approx(0.534, abs=0.001), marks=[pytest.mark.private_access]),

    # Public
    pytest.param("googlenet", "MajajHong2015public.IT-pls", approx(0.539, abs=0.001)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True,
    )
    assert actual_score == expected_score
