import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('vit_huge_clip224_laion_in1k')
    assert model.identifier == 'vit_huge_clip224_laion_in1k'
