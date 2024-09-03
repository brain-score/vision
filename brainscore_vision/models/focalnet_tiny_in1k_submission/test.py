import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('focalnet_tiny_lrf_in1k')
    assert model.identifier == 'focalnet_tiny_lrf_in1k'