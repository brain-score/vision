import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('CORnet-S')
    assert model.identifier == 'CORnet-S'