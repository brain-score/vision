from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['vgg11'] = lambda: ModelCommitment(
    identifier='vgg11',
    activations_model=get_model("vgg11"),
    layers=get_layers('vgg11'))

model_registry['vgg11_bn'] = lambda: ModelCommitment(
    identifier='vgg11_bn',
    activations_model=get_model("vgg11_bn"),
    layers=get_layers('vgg11_bn'))

model_registry['vgg13'] = lambda: ModelCommitment(
    identifier='vgg13',
    activations_model=get_model("vgg13"),
    layers=get_layers('vgg13'))

model_registry['vgg13_bn'] = lambda: ModelCommitment(
    identifier='vgg13_bn',
    activations_model=get_model("vgg13_bn"),
    layers=get_layers('vgg13_bn'))

model_registry['vgg16'] = lambda: ModelCommitment(
    identifier='vgg16',
    activations_model=get_model("vgg16"),
    layers=get_layers('vgg16'))

model_registry['vgg16_bn'] = lambda: ModelCommitment(
    identifier='vgg16_bn',
    activations_model=get_model("vgg16_bn"),
    layers=get_layers('vgg16_bn'))

model_registry['vgg19'] = lambda: ModelCommitment(
    identifier='vgg19',
    activations_model=get_model("vgg19"),
    layers=get_layers('vgg19'))

model_registry['vgg19_bn'] = lambda: ModelCommitment(
    identifier='vgg19_bn',
    activations_model=get_model("vgg19_bn"),
    layers=get_layers('vgg19_bn'))
