import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet-18_test_m')
    assert model.identifier == 'resnet-18_test_m'