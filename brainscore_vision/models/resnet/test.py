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
    pytest.param("resnet-18", "MajajHong2015.IT-pls", approx(0.527, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("resnet-34", "MajajHong2015.IT-pls", approx(0.516, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("resnet-50", "MajajHong2015.IT-pls", approx(0.523, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("resnet-101", "MajajHong2015.IT-pls", approx(0.51, abs=0.0005), marks=[pytest.mark.private_access]),

    # Public
    pytest.param("resnet-18", "MajajHong2015public.IT-pls", approx(0.548, abs=0.0005)),
    pytest.param("resnet-34", "MajajHong2015public.IT-pls", approx(0.543, abs=0.0005)),
    pytest.param("resnet-50", "MajajHong2015public.IT-pls", approx(0.537, abs=0.0005)),
    pytest.param("resnet-101", "MajajHong2015public.IT-pls", approx(0.528, abs=0.0005)),

])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
