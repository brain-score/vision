import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything
seed_everything(42)

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score",
    [
        ("neg_samples_e2e", "MajajHong2015.IT-pls", approx(0.4413, abs=0.001)),
    ],
)
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True,
    )

    assert actual_score == expected_score
