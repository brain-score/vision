import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('nasnet_large')
    assert model.identifier == 'nasnet_large'