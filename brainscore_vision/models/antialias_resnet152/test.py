import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('antialias-resnet152')
    assert model.identifier == 'antialias-resnet152'