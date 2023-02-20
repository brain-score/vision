from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['texture_shape_resnet50_trained_on_SIN'] = ModelCommitment(identifier='texture_shape_resnet50_trained_on_SIN', activations_model=get_model('texture_shape_resnet50_trained_on_SIN'), layers=get_layers('texture_shape_resnet50_trained_on_SIN'))
