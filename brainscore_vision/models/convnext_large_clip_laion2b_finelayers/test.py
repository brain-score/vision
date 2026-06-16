import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('convnext_large_clip_laion2b_finelayers')
    assert model.identifier == 'convnext_large_clip_laion2b_finelayers'
