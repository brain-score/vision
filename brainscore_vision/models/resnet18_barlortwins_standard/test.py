import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet18_barlortwins:standard_in1k_ba1024_ep100')
    assert model.identifier == 'resnet18_barlortwins:standard_in1k_ba1024_ep100'
