import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@pytest.mark.slow
@pytest.mark.travis_slow
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    # Private
    pytest.param("cv_18_dagger_408_pretrained", "MajajHong2015.IT-pls", approx(0.5126, abs=0.0005), marks=[pytest.mark.private_access]),

    # Public
    pytest.param("cv_18_dagger_408_pretrained", "MajajHong2015public.IT-pls", approx(0.5362, abs=0.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=False)
    assert actual_score == expected_score
