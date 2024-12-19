from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['evresnet_50_1'] = lambda: ModelCommitment(
    identifier='evresnet_50_1',
    activations_model=get_model('evresnet_50_1'),
    layers=get_layers('evresnet_50_1'),
    behavioral_readout_layer='model.5',
    region_layer_map={'V1': 'voneblock'},
    visual_degrees=7,
    )
