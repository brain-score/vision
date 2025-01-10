from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['voneresnet_50_no_weight'] = lambda: ModelCommitment(
    identifier='voneresnet_50_no_weight',
    activations_model=get_model('voneresnet_50_no_weight'),
    layers=get_layers('voneresnet_50_no_weight'),
    region_layer_map={'V1': 'voneblock'},
    visual_degrees=7,
    )
