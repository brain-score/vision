import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet_SIN_IN_FT_IN')
    assert model.identifier == 'resnet_SIN_IN_FT_IN'