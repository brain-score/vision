import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything
seed_everything(42)

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("vgg11",  "MajajHong2015public.IT-pls", approx(0.545, abs=0.001)),
    ("vgg11_bn",  "MajajHong2015public.IT-pls", approx(0.534, abs=0.001)),
    ("vgg13",  "MajajHong2015public.IT-pls", approx(0.534, abs=0.001)),
    ("vgg13_bn",  "MajajHong2015public.IT-pls", approx(0.526, abs=0.001)),
    ("vgg16",  "MajajHong2015public.IT-pls", approx(0.54, abs=0.001)),
    ("vgg16_bn",  "MajajHong2015public.IT-pls", approx(0.531, abs=0.001)),
    ("vgg19",  "MajajHong2015public.IT-pls", approx(0.553, abs=0.001)),
    ("vgg19_bn",  "MajajHong2015public.IT-pls", approx(0.497, abs=0.001)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
