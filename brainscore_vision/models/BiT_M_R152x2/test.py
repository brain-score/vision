import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('BiT-M-R152x2')
    assert model.identifier == 'BiT-M-R152x2'