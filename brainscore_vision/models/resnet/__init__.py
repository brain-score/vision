from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-18'] = lambda: ModelCommitment(
    identifier='resnet-18',
    activations_model=get_model('resnet-18'),
    layers=get_layers('resnet-18'))

model_registry['resnet-34'] = lambda: ModelCommitment(
    identifier='resnet-34',
    activations_model=get_model('resnet-34'),
    layers=get_layers('resnet-34'))

model_registry['resnet-50'] = lambda: ModelCommitment(
    identifier='resnet-50',
    activations_model=get_model('resnet-50'),
    layers=get_layers('resnet-50'))

model_registry['resnet-101'] = lambda: ModelCommitment(
    identifier='resnet-101',
    activations_model=get_model('resnet-101'),
    layers=get_layers('resnet-101'))

model_registry['resnet-152'] = lambda: ModelCommitment(
    identifier='resnet-152',
    activations_model=get_model('resnet-152'),
    layers=get_layers('resnet-152'))
