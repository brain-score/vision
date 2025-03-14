import pytest
import brainscore_vision

5@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('dinov2_base')
    assert model.identifier == 'dinov2_base'
    