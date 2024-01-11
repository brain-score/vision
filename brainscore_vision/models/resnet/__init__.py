from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['resnet-18'] = lambda: ModelCommitment(
    identifier='resnet-18',
    activations_model=get_model('resnet-18'),
    layers=LAYERS['resnet-18'])

model_registry['resnet-34'] = lambda: ModelCommitment(
    identifier='resnet-34',
    activations_model=get_model('resnet-34'),
    layers=LAYERS['resnet-34'])

model_registry['resnet-50'] = lambda: ModelCommitment(
    identifier='resnet-50',
    activations_model=get_model('resnet-50'),
    layers=LAYERS['resnet-50'])

model_registry['resnet-101'] = lambda: ModelCommitment(
    identifier='resnet-101',
    activations_model=get_model('resnet-101'),
    layers=LAYERS['resnet-101'])
