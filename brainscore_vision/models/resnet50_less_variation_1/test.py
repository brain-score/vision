
import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50_less_variation_iteration=1')
    assert model.identifier == 'resnet50_less_variation_iteration=1'
