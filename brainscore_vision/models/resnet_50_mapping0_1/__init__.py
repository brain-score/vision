from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet_50_mapping0_1'] = lambda: ModelCommitment(
    identifier='resnet_50_mapping0_1',
    activations_model=get_model('resnet_50_mapping0_1'),
    layers=get_layers('resnet_50_mapping0_1'),
    region_layer_map={'V1': 'model.maxpool', 'V2':'model.layer2', 'V4':'model.layer3', 'IT':'model.layer4'}
    )