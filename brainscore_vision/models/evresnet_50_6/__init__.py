from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['evresnet_50_6'] = lambda: ModelCommitment(
    identifier='evresnet_50_6',
    activations_model=get_model('evresnet_50_6'),
    layers=get_layers('evresnet_50_6'),
    behavioral_readout_layer='model.avgpool',
    region_layer_map={'V1': 'voneblock', 'V2': 'model.layer3.1', 'V4': 'model.layer2.1', 'IT': 'model.layer4.0'},
    visual_degrees=7
    )