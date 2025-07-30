import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('evresnet_50_1')
    assert model.identifier == 'evresnet_50_1'