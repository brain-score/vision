import pytest
from pytest import approx

from brainscore_vision import score


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("alexnet", "MajajHong2015.IT-pls", approx(0.508, abs=0.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
