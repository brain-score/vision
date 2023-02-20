from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['my-resnet-img-model'] = ModelCommitment(identifier='my-resnet-img-model', activations_model=get_model('my-resnet-img-model'), layers=get_layers('my-resnet-img-model'))
