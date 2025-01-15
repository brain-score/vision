import brainscore_vision
import pytest



@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('efficientnet-b7')
    assert model.identifier == 'efficientnet-b7'