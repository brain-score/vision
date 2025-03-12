import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('mobilenet_v2_1_0_224')
    assert model.identifier == 'mobilenet_v2_1_0_224'