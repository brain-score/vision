import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('efficientnet_b0')
    assert model.identifier == 'efficientnet_b0'