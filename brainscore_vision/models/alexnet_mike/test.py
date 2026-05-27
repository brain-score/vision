import pytest
from pytest import approx

from brainscore_vision import score


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("alexnet", "MajajHong2015.IT-pls", approx(0.417, abs=0.001)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    assert True
