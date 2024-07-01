import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('effnetb1_cutmix_augmix_sam_e1_5avg_424x377')
    assert model.identifier == 'effnetb1_cutmix_augmix_sam_e1_5avg_424x377'