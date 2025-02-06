import pytest

from brainscore_vision import load_model


model_list = [
    "SeLaVi-Kinetics400",
    "SeLaVi-Kinetics-Sound",
    "SeLaVi-VGG-Sound",
    "SeLaVi-AVE"
]

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier", model_list)
def test_load(model_identifier):
    model = load_model(model_identifier)
    assert model is not None