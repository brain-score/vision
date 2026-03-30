import pytest
import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model('resnet18_fair')
    assert model.identifier == 'resnet18_fair'


def test_has_layers():
    model = brainscore_vision.load_model('resnet18_fair')
    assert 'layer1' in model.layers
    assert 'layer4' in model.layers
