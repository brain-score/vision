import pytest

from brainscore_vision import load_model


model_list = [
    "VideoMAE-V2-B",
    "VideoMAE-V2-G",
]

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier", model_list)
def test_load(model_identifier):
    model = load_model(model_identifier)
    assert model is not None