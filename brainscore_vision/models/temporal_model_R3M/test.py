import pytest

from brainscore_vision import load_model


model_list = [
    "R3M-ResNet18-Temporal",
    "R3M-ResNet34-Temporal",
    "R3M-ResNet50-Temporal",
]

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier", model_list)
def test_load(model_identifier):
    model = load_model(model_identifier)
    assert model is not None