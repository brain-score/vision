import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('voneresnet-50-non_stochastic')
    assert model.identifier == 'voneresnet-50-non_stochastic'