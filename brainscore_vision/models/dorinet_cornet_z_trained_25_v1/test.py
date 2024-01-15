import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything
seed_everything(42)

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("dorinet_cornet_z_trained_25_v1",  "MajajHong2015public.IT-pls", approx(0.34, abs=0.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
