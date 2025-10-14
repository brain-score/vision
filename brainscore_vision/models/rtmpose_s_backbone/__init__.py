from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['rtmpose_s_backbone'] = lambda: ModelCommitment(
    identifier='rtmpose_s_backbone',
    activations_model=get_model(),
    layers=get_layers(),
)
