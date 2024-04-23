import pytest
import brainscore_vision


@pytest.mark.travis_slow
def test_has_identifier():
    model = brainscore_vision.load_model('resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234')
    assert model.identifier == 'resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234'