import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50_dvd_pure10deg_wzc_20260913_seed5_epoch90')
    assert model.identifier == 'resnet50_dvd_pure10deg_wzc_20260913_seed5_epoch90'
