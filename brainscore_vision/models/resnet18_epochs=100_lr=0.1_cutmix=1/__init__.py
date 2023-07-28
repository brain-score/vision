from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['img100-resnet-lr0.1-100e-cutmix1'] = ModelCommitment(identifier='img100-resnet-lr0.1-100e-cutmix1', activations_model=get_model('img100-resnet-lr0.1-100e-cutmix1'), layers=get_layers('img100-resnet-lr0.1-100e-cutmix1'))
