from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['evresnet_50_4'] = lambda: ModelCommitment(
    identifier='evresnet_50_4',
    activations_model=get_model('evresnet_50_4'),
    layers=get_layers('evresnet_50_4'),
    behavioral_readout_layer='model.5',
    region_layer_map={"V1": "voneblock", "V2": "model.2.1", "V4": "model.1.1", "IT": "model.3.0"},
    visual_degrees=7,
    )