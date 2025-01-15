import brainscore_vision
import pytest



@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('shufflenet_v2_x1_0')
    assert model.identifier == 'shufflenet_v2_x1_0'