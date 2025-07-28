import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('abl_mcells_evresnet_50_457_0')
    assert model.identifier == 'abl_mcells_evresnet_50_457_0'