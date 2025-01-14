import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('densenet-201')
    assert model.identifier == 'densenet-201'
