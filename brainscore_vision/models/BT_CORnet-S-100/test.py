import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('BT_CORnet-S-100')
    assert model.identifier == 'BT_CORnet-S-100'