import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50_ewconplus_wzc_seed7_2')
    assert model.identifier == 'resnet50_ewconplus_wzc_seed7_2'
