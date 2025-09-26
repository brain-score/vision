import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('compact_vgg19_IT')
    assert model.identifier == 'compact_vgg19_IT'
