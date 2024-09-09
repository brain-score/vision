import brainscore_vision
import pytest



@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('omnivore_swinS')
    assert model.identifier == 'omnivore_swinS'