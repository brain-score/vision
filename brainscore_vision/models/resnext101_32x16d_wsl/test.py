import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnext101_32x16d_wsl')
    assert model.identifier == 'resnext101_32x16d_wsl'