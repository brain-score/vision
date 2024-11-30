
import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('vgg16_less_variation_iteration=2')
    assert model.identifier == 'vgg16_less_variation_iteration=2'
