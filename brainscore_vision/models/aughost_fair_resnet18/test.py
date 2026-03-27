import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model('aughost_fair_resnet18')
    assert model.identifier == 'aughost_fair_resnet18'
