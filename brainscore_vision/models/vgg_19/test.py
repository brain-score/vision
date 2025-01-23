import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('vgg_19')
    assert model.identifier == 'vgg_19'
