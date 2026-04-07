import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('convnext_base_fb_in22k_ft_in1k')
    assert model.identifier == 'convnext_base_fb_in22k_ft_in1k'
