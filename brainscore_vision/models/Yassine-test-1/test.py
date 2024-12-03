import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('Yassine-test-1')
    assert model.identifier == "Yassine-test-1"
