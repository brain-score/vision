import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model("resnet18_blur")
    assert model.identifier == "resnet18_blur"
