# __init__.py - Model registration for hk_model_2 (SwAV)
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# hk_model_2: Self-supervised ResNet-50 (SwAV) registration
model_registry['hk_model_2'] = lambda: ModelCommitment(
    identifier='hk_model_2',
    activations_model=get_model('hk_model_2'),
    layers=get_layers('hk_model_2')
)
