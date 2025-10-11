# test.py - hk_model_1 테스트
import pytest
import brainscore_vision

def test_hk_model_1_has_identifier():
    model = brainscore_vision.load_model('hk_model_1')
    assert model.identifier == 'hk_model_1'

def test_hk_model_1_has_activations_model():
    model = brainscore_vision.load_model('hk_model_1')
    assert hasattr(model, 'activations_model')

def test_hk_model_1_layers():
    from brainscore_vision.models.hk_model_1 import get_layers
    layers = get_layers('hk_model_1')
    assert len(layers) > 0
    assert 'layer4' in layers