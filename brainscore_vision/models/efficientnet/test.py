import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score",
    [
        ("efficientnet_b0", "MajajHong2015public.IT-pls", approx(0.549, abs=0.001)),
        ("efficientnet_b1", "MajajHong2015public.IT-pls", approx(0.5576, abs=0.001)),
        ("efficientnet_b2", "MajajHong2015public.IT-pls", approx(0.546, abs=0.001)),
        ("efficientnet_b3", "MajajHong2015public.IT-pls", approx(0.540, abs=0.001)),
        ("efficientnet_b4", "MajajHong2015public.IT-pls", approx(0.528, abs=0.001)),
        ("efficientnet_b5", "MajajHong2015public.IT-pls", approx(0.551, abs=0.001)),
        ("efficientnet_b6", "MajajHong2015public.IT-pls", approx(0.543, abs=0.001)),
        ("efficientnet_b7", "MajajHong2015public.IT-pls", approx(0.545, abs=0.001)),
    ],
)
def test_score(model_identifier, benchmark_identifier, expected_score):
    seed_everything(42)
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True,
    )
    assert actual_score == expected_score
