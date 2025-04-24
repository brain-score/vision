import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('kap-l1-2')
    assert model.identifier == 'kap-l1-2'
