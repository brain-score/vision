import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('Res2Net50_26w_4s')
    assert model.identifier == 'Res2Net50_26w_4s'
