from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['barlow-twins-resnet50'] = lambda: ModelCommitment(identifier='barlow-twins-resnet50', activations_model=get_model('barlow-twins-resnet50'), layers=LAYERS)
