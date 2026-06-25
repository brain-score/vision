import pytest, brainscore_vision
@pytest.mark.travis_slow
def test_has_identifier():
    assert brainscore_vision.load_model('convnext_xxlarge_clip').identifier == 'convnext_xxlarge_clip'
