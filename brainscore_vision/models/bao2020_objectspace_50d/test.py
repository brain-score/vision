import pytest
from pytest import approx

from brainscore_vision import score


@pytest.mark.slow
@pytest.mark.travis_slow
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("bao2020_objectspace_50d", "MajajHong2015public.IT-pls", approx(0.3168, abs=0.005)),
    ("bao2020_objectspace_50d", "MajajHong2015public.V4-pls", approx(0.1776, abs=0.005)),
    pytest.param("bao2020_objectspace_50d", "MajajHong2015.IT-pls", approx(0.3233, abs=0.005),
                 marks=[pytest.mark.private_access]),
    pytest.param("bao2020_objectspace_50d", "MajajHong2015.V4-pls", approx(0.1738, abs=0.005),
                 marks=[pytest.mark.private_access]),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=False)
    assert actual_score == expected_score
