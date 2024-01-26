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
        ("vit_b_16", "MajajHong2015public.IT-pls", approx(0.489, abs=0.001)),
        ("vit_b_32", "MajajHong2015public.IT-pls", approx(0.499, abs=0.001)),
        ("vit_l_16", "MajajHong2015public.IT-pls", approx(0.543, abs=0.001)),
        ("vit_l_32", "MajajHong2015public.IT-pls", approx(0.51, abs=0.001)),
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
