import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('pm1_evresnet_50_457_0')
    assert model.identifier == 'pm1_evresnet_50_457_0'