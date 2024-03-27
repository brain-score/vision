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
    pytest.param("efficientnet_b0", "MajajHong2015.IT-pls", approx(0.526, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("efficientnet_b1", "MajajHong2015.IT-pls", approx(0.538, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("efficientnet_b2", "MajajHong2015.IT-pls", approx(0.529, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("efficientnet_b3", "MajajHong2015.IT-pls", approx(0.511, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("efficientnet_b4", "MajajHong2015.IT-pls", approx(0.514, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("efficientnet_b5", "MajajHong2015.IT-pls", approx(0.528, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("efficientnet_b6", "MajajHong2015.IT-pls", approx(0.526, abs=0.0005), marks=[pytest.mark.private_access]),
    pytest.param("efficientnet_b7", "MajajHong2015.IT-pls", approx(0.535, abs=0.0005), marks=[pytest.mark.private_access]),

    # Public
    pytest.param("efficientnet_b0", "MajajHong2015public.IT-pls", approx(0.549, abs=0.0005)),
    pytest.param("efficientnet_b1", "MajajHong2015public.IT-pls", approx(0.558, abs=0.0005)),
    pytest.param("efficientnet_b2", "MajajHong2015public.IT-pls", approx(0.546, abs=0.0005)),
    pytest.param("efficientnet_b3", "MajajHong2015public.IT-pls", approx(0.540, abs=0.0005)),
    pytest.param("efficientnet_b4", "MajajHong2015public.IT-pls", approx(0.528, abs=0.0005)),
    pytest.param("efficientnet_b5", "MajajHong2015public.IT-pls", approx(0.551, abs=0.0005)),
    pytest.param("efficientnet_b6", "MajajHong2015public.IT-pls", approx(0.543, abs=0.0005)),
    pytest.param("efficientnet_b7", "MajajHong2015public.IT-pls", approx(0.545, abs=0.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True,
    )
    assert actual_score == expected_score
