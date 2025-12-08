import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('compact_vgg19_V4')
    assert model.identifier == 'compact_vgg19_V4'
