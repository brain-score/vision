from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['densenet_201_pytorch'] = lambda: ModelCommitment(identifier='densenet_201_pytorch',
                                                               activations_model=get_model(),
                                                               layers=get_layers())