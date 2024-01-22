from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['resnet-lr0.01-500c'] = lambda: ModelCommitment(
    identifier='resnet-lr0.01-500c',
    activations_model=get_model(),
    layers=LAYERS)
