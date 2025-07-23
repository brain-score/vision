import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('voneresnet-50')
    assert model.identifier == 'voneresnet-50'