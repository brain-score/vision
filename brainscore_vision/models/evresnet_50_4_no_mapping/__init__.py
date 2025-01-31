from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['evresnet_50_4_no_mapping'] = lambda: ModelCommitment(
    identifier='evresnet_50_4_no_mapping',
    activations_model=get_model('evresnet_50_4_no_mapping'),
    layers=get_layers('evresnet_50_4_no_mapping'),
    visual_degrees=7
    )