import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('effnetb1_cutmixpatch_SAM_robust32_avge6e8e9e10_manylayers_324x288')
    assert model.identifier == 'effnetb1_cutmixpatch_SAM_robust32_avge6e8e9e10_manylayers_324x288'