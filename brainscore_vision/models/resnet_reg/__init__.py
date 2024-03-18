from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['resnet18_reg'] = lambda: ModelCommitment(
    identifier='resnet18_reg',
    activations_model=get_model('resnet18_reg'),
    layers=LAYERS['resnet18_reg'])

model_registry['resnet50_reg'] = lambda: ModelCommitment(
    identifier='resnet50_reg',
    activations_model=get_model('resnet50_reg'),
    layers=LAYERS['resnet50_reg'])
