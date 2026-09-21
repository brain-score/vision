import pytest

from brainscore_vision import load_model


@pytest.mark.private_access
@pytest.mark.memory_intense
def test_model_loads():
    model = load_model("vjepa2-vitg")
    assert model is not None
