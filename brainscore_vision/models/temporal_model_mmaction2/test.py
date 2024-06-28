import pytest

from brainscore_vision import load_model


model_list = [
    "I3D",
    "I3D-nonlocal",
    "SlowFast",
    "X3D",
    "TimeSformer",
    "VideoSwin-B",
    "VideoSwin-L",
    "UniFormer-V1",
    "UniFormer-V2-B",
    "UniFormer-V2-L",
]

@pytest.mark.private_access
@pytest.mark.memory_intense
@pytest.mark.parametrize("model_identifier", model_list)
def test_load(model_identifier):
    model = load_model(model_identifier)
    assert model is not None