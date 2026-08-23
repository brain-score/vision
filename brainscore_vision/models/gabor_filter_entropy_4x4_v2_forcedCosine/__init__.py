from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers, LAYER_NAME


_REGION_LAYER_MAP = {
    'V1': LAYER_NAME,
    'V2': LAYER_NAME,
    'V4': LAYER_NAME,
    'IT': LAYER_NAME,
}


model_registry['gabor_filter_entropy_4x4_v2_forcedCosine'] = lambda: ModelCommitment(
    identifier='gabor_filter_entropy_4x4_v2_forcedCosine',
    activations_model=get_model('gabor_filter_entropy_4x4_v2_forcedCosine'),
    layers=get_layers('gabor_filter_entropy_4x4_v2_forcedCosine'),
    region_layer_map=_REGION_LAYER_MAP)
