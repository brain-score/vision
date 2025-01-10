import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('imagenet_l2_3_0')
    assert model.identifier == 'imagenet_l2_3_0'