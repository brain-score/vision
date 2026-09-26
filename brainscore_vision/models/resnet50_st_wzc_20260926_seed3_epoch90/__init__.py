from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

REGION_LAYER_MAP = {'V1': 'layer2.1.conv1', 'V2': 'layer3.3.conv2', 'V4': 'layer4.0.conv2', 'IT': 'layer4.1.conv1'}

model_registry['resnet50_st_wzc_20260926_seed3_epoch90'] = lambda: ModelCommitment(
    identifier='resnet50_st_wzc_20260926_seed3_epoch90',
    activations_model=get_model('resnet50_st_wzc_20260926_seed3_epoch90'),
    layers=get_layers('resnet50_st_wzc_20260926_seed3_epoch90'),
    region_layer_map=REGION_LAYER_MAP,
)
