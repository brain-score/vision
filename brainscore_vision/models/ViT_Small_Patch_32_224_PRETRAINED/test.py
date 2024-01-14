# Left empty as part of 2023 models migration
import pytest
from pytest import approx

from brainscore_vision import score
from brainscore_vision.utils import seed_everything
seed_everything(42)

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier, benchmark_identifier, expected_score", [
    ("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-IT",  "MajajHong2015public.IT-pls", approx(0.4792, abs=0.0005)),
    ("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V1",  "MajajHong2015public.IT-pls", approx(0.3563, abs=0.0005)),
    ("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V2",  "MajajHong2015public.IT-pls", approx(0.4748, abs=0.0005)),
    ("ViT-Small-Patch-32-224-PRETRAINED-INPUT-SIZE-256-CROP-SIZE-224-V4",  "MajajHong2015public.IT-pls", approx(0.3787, abs=0.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         conda_active=True)
    assert actual_score == expected_score
