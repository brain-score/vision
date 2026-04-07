import pytest
import brainscore_vision

@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('deit3_base_patch16_224_fb_in1k')
    assert model.identifier == 'deit3_base_patch16_224_fb_in1k'
