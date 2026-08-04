import pytest, brainscore_vision
@pytest.mark.travis_slow
def test_has_identifier():
    assert brainscore_vision.load_model('vit_huge_clip336_laion').identifier == 'vit_huge_clip336_laion'
