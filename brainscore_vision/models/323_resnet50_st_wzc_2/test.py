import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('323_resnet50_st_wzc_2')
    assert model.identifier == '323_resnet50_st_wzc_2'
