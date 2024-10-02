import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet-152_v2_pytorch')
    assert model.identifier == 'resnet-152_v2_pytorch'