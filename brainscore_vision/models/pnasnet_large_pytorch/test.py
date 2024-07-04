import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('pnasnet_large_pytorch')
    assert model.identifier == 'pnasnet_large_pytorch'