# test.py - hk_model_2 tests (Self-supervised SwAV)
import pytest
import brainscore_vision

def test_hk_model_2_has_identifier():
    model = brainscore_vision.load_model('hk_model_2')
    assert model.identifier == 'hk_model_2'

def test_hk_model_2_has_activations_model():
    model = brainscore_vision.load_model('hk_model_2')
    assert hasattr(model, 'activations_model')

def test_hk_model_2_layers():
    from brainscore_vision.models.hk_model_2 import get_layers
    layers = get_layers('hk_model_2')
    assert len(layers) == 6
    assert 'conv1' in layers
    assert 'layer1' in layers
    assert 'layer2' in layers
    assert 'layer3' in layers
    assert 'layer4' in layers
    assert 'fc' in layers
