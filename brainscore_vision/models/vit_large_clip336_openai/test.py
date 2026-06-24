import pytest, brainscore_vision
@pytest.mark.travis_slow
def test_has_identifier():
    assert brainscore_vision.load_model('vit_large_clip336_openai').identifier == 'vit_large_clip336_openai'
