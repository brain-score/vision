import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('antialiased-rnext101_32x8d')
    assert model.identifier == 'antialiased-rnext101_32x8d'