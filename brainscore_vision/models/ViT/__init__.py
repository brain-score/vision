from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['B_16_imagenet1k'] = ModelCommitment(identifier='B_16_imagenet1k', activations_model=get_model('B_16_imagenet1k'), layers=get_layers('B_16_imagenet1k'))
model_registry['B_32_imagenet1k'] = ModelCommitment(identifier='B_32_imagenet1k', activations_model=get_model('B_32_imagenet1k'), layers=get_layers('B_32_imagenet1k'))
model_registry['L_16_imagenet1k'] = ModelCommitment(identifier='L_16_imagenet1k', activations_model=get_model('L_16_imagenet1k'), layers=get_layers('L_16_imagenet1k'))
model_registry['L_32_imagenet1k'] = ModelCommitment(identifier='L_32_imagenet1k', activations_model=get_model('L_32_imagenet1k'), layers=get_layers('L_32_imagenet1k'))
model_registry['B_16'] = ModelCommitment(identifier='B_16', activations_model=get_model('B_16'), layers=get_layers('B_16'))
model_registry['B_32'] = ModelCommitment(identifier='B_32', activations_model=get_model('B_32'), layers=get_layers('B_32'))
model_registry['L_32'] = ModelCommitment(identifier='L_32', activations_model=get_model('L_32'), layers=get_layers('L_32'))
