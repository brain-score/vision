from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['beit3_large_patch16_224_in1k'] = lambda: ModelCommitment(
    identifier='beit3_large_patch16_224_in1k',
    activations_model=get_model('beit3_large_patch16_224_in1k'),
    layers=get_layers('beit3_large_patch16_224_in1k'),
    behavioral_readout_layer='fc_norm',
    visual_degrees=8)
