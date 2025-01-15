import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('voneresnet_50_no_weight')
    assert model.identifier == 'voneresnet_50_no_weight'