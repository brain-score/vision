import pytest

from brainscore_vision import load_model


model_list = [
    "r3d_18",
    "r2plus1d_18",
    "mc3_18",
    "s3d",
    "mvit_v1_b",
    "mvit_v2_s",
]

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier", model_list)
def test_load(model_identifier):
    model = load_model(model_identifier)
    assert model is not None