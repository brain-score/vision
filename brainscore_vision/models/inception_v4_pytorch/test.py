import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('inception_v4_pytorch')
    assert model.identifier == 'inception_v4_pytorch'