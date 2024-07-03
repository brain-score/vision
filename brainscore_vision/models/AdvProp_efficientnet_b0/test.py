import brainscore_vision
import pytest



@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('AdvProp_efficientnet-b0')
    assert model.identifier == 'AdvProp_efficientnet-b0'