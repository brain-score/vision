from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['evresnet_50_2'] = lambda: ModelCommitment(
    identifier='evresnet_50_2',
    activations_model=get_model('evresnet_50_2'),
    layers=get_layers('evresnet_50_2'),
    behavioral_readout_layer='model.5',
    region_layer_map={'V1': 'vonenet'},
    visual_degrees=7,
    )