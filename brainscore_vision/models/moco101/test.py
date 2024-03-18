import logging
import sys

import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
seed_everything(42)

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score",
    [
        ("moco_101_20", "MajajHong2015public.IT-pls", approx(0.576, abs=0.001)),
        ("moco_101_30", "MajajHong2015public.IT-pls", approx(0.576, abs=0.001)),
        ("moco_101_40", "MajajHong2015public.IT-pls", approx(0.586, abs=0.001)),
        ("moco_101_50", "MajajHong2015public.IT-pls", approx(0.585, abs=0.001)),
        ("moco_101_60", "MajajHong2015public.IT-pls", approx(0.581, abs=0.001)),
        ("moco_101_70", "MajajHong2015public.IT-pls", approx(0.587, abs=0.001)),
    ],
)
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True,
    )
    assert actual_score == expected_score
