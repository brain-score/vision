from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# Layer -> brain region mapping (standard ResNet hierarchy convention)
REGION_LAYER_MAP = {
    'V1': 'layer1',
    'V2': 'layer2',
    'V4': 'layer3',
    'IT': 'layer4',
}

model_registry['resnet18_fair'] = lambda: ModelCommitment(
    identifier='resnet18_fair',
    activations_model=get_model('resnet18_fair'),
    layers=get_layers('resnet18_fair'),
    region_layer_map=REGION_LAYER_MAP,
)
