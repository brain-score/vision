import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('vit_large_clip224_openai_in1k_lsearch')
    assert model.identifier == 'vit_large_clip224_openai_in1k_lsearch'
