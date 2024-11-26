import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('blt-vs')
    assert model.identifier == 'blt-vs'