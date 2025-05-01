import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('simple_spiking_model')
    assert model.identifier == 'simple_spiking_model'