import pytest
from pytest import approx

from brainscore_vision import score


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("resnet-18", "MajajHong2015public.IT-pls", approx(0.540, abs=0.001)),
    ("resnet-34", "MajajHong2015public.IT-pls", approx(0.510, abs=0.001)),
    ("resnet-50", "MajajHong2015public.IT-pls", approx(0.527, abs=0.001)),
    ("resnet-101", "MajajHong2015public.IT-pls", approx(0.538, abs=0.001)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
