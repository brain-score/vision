import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('AdvProp_efficientnet-b7')
    assert model.identifier == 'AdvProp_efficientnet-b7'