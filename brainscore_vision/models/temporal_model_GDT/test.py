import pytest

from brainscore_vision import load_model


model_list = [
    "GDT-Kinetics400",
    "GDT-HowTo100M",
    "GDT-IG65M",
]

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier", model_list)
def test_load(model_identifier):
    model = load_model(model_identifier)
    assert model is not None