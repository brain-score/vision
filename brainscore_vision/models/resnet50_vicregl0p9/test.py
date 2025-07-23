import brainscore_vision
import pytest



@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50-vicregl0p9')
    assert model.identifier == 'resnet50-vicregl0p9'