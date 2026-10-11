import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('beit3_large_patch16_224_in1k')
    assert model.identifier == 'beit3_large_patch16_224_in1k'
