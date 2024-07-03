import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('cvt_cvt-w24-384-in22k_finetuned-in1k_4')
    assert model.identifier == 'cvt_cvt-w24-384-in22k_finetuned-in1k_4'