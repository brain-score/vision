import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('dinov2_vitl14_reg_lc')
    assert model.identifier == 'dinov2_vitl14_reg_lc'
