from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['voneresnet_50_3'] = lambda: ModelCommitment(
    identifier='voneresnet_50_3',
    activations_model=get_model('voneresnet_50_3'),
    layers=get_layers('voneresnet_50_3'),
    region_layer_map={"V1": "voneblock", "V2": "model.2.1", "V4": "model.1.1", "IT": "model.3.0"},
    visual_degrees=7,
    )
