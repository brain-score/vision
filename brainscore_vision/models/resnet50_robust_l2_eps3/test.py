import brainscore_vision
import pytest


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50_robust_l2_eps3')
    assert model.identifier == 'resnet50_robust_l2_eps3'