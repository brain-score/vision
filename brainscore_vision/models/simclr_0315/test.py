import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('simclr_0315')
    assert model.identifier == 'simclr_0315'
