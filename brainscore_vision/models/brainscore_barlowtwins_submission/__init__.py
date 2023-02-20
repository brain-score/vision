from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['barlow-twins-resnet50'] = ModelCommitment(identifier='barlow-twins-resnet50', activations_model=get_model('barlow-twins-resnet50'), layers=get_layers('barlow-twins-resnet50'))
