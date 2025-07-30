import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('inception_v1')
    assert model.identifier == 'inception_v1'
