import brainscore_vision
import pytest



@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('mobilenet_v2_0_5_224')
    assert model.identifier == 'mobilenet_v2_0_5_224'