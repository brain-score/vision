import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet_152_v1')
    assert model.identifier == 'resnet_152_v1'