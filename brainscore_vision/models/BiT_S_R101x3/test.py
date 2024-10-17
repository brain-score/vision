import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('BiT-S-R101x3')
    assert model.identifier == 'BiT-S-R101x3'