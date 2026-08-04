import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('eva02_large_in22k')
    assert model.identifier == 'eva02_large_in22k'
