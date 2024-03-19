from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
# from .model import get_model, get_layers
from .model import get_model, LAYERS

model_registry['dinov2'] = lambda: ModelCommitment(
    identifier='dinov2',
    activations_model=get_model("dinov2"),
    layers=LAYERS)
