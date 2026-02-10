import pytest
from pytest import approx

import brainscore_vision
from brainscore_vision import score


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('pnasnet_large_half')
    assert model.identifier == 'pnasnet_large_half'


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("pnasnet_large_half", "SanghaviMurty2020.V4-pls", approx(0.206, abs=0.005)),
    ("pnasnet_large_half", "FreemanZiemba2013.V1-pls", approx(0.226, abs=0.005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
