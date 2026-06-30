import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('Zejin_Lu_dvd_p_imagenet_resnet50')
    assert model.identifier == 'Zejin_Lu_dvd_p_imagenet_resnet50'
