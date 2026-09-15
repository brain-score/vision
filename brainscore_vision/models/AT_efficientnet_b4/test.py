import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('AT_efficientnet-b4')
    assert model.identifier == 'AT_efficientnet-b4'