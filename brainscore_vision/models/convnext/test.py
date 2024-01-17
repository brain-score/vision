import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything

seed_everything(42)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score",
    [
        ("convnext_tiny", "MajajHong2015public.IT-pls", approx(0.562, abs=0.001)),
        ("convnext_small", "MajajHong2015public.IT-pls", approx(0.554, abs=0.001)),
        ("convnext_base", "MajajHong2015public.IT-pls", approx(0.553, abs=0.001)),
        ("convnext_large", "MajajHong2015public.IT-pls", approx(0.557, abs=0.001)),
    ],
)
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True,
    )
    assert actual_score == expected_score
