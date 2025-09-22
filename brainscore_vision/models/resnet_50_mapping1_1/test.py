import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet_50_mapping1_1')
    assert model.identifier == 'resnet_50_mapping1_1'