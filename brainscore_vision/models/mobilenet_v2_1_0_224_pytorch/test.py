import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('mobilenet_v2_1-0_224_pytorch')
    assert model.identifier == 'mobilenet_v2_1-0_224_pytorch'