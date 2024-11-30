
import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model(f'vgg16_less_variation_iteration=1')
    assert model.identifier == f'vgg16_less_variation_iteration=1'
