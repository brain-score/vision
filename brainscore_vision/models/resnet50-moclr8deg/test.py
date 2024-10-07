import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50-moclr8deg')
    assert model.identifier == 'resnet50-moclr8deg'