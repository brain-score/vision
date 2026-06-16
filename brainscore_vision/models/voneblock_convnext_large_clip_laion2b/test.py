import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('voneblock_convnext_large_clip_laion2b')
    assert model.identifier == 'voneblock_convnext_large_clip_laion2b'
