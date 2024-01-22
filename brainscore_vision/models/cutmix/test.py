import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything
seed_everything(42)


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("r50-e10-cut1", "MajajHong2015public.IT-pls", approx(0.5266, abs=0.0005)),
    ("r50-e20-cut1", "MajajHong2015public.IT-pls", approx(0.5304, abs=0.0005)),
    ("r50-e35-cut1", "MajajHong2015public.IT-pls", approx(0.5432, abs=0.0005)),
    ("r50-e50-cut1", "MajajHong2015public.IT-pls", approx(0.5452, abs=0.0005)),
    ("imgnfull-e45-cut1", "MajajHong2015public.IT-pls", approx(0.5315, abs=0.0005)),
    ("imgnfull-e60-cut1", "MajajHong2015public.IT-pls", approx(0.5302, abs=0.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
