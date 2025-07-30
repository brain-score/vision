import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('abl_light_evresnet_50_457_0')
    assert model.identifier == 'abl_light_evresnet_50_457_0'