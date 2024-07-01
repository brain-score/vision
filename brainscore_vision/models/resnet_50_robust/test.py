import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet-50-robust')
    assert model.identifier == 'resnet-50-robust'