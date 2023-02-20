from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-10-two-blocks'] = ModelCommitment(identifier='resnet-10-two-blocks', activations_model=get_model('resnet-10-two-blocks'), layers=get_layers('resnet-10-two-blocks'))
model_registry['resnet-10-two-blocks-LC'] = ModelCommitment(identifier='resnet-10-two-blocks-LC', activations_model=get_model('resnet-10-two-blocks-LC'), layers=get_layers('resnet-10-two-blocks-LC'))
model_registry['resnet-10m-two-blocks'] = ModelCommitment(identifier='resnet-10m-two-blocks', activations_model=get_model('resnet-10m-two-blocks'), layers=get_layers('resnet-10m-two-blocks'))
model_registry['resnet-10m-two-blocks-LC'] = ModelCommitment(identifier='resnet-10m-two-blocks-LC', activations_model=get_model('resnet-10m-two-blocks-LC'), layers=get_layers('resnet-10m-two-blocks-LC'))
