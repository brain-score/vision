import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('cvt_cvt-21-384-in1k_4')
    assert model.identifier == 'cvt_cvt-21-384-in1k_4'