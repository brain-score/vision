from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
# from .model import get_model, get_layers
from .model import get_model, LAYERS

model_registry['dinov2_all_layers'] = lambda: ModelCommitment(
    identifier='dinov2_all_layers',
    activations_model=get_model("dinov2_all_layers"),
    layers=LAYERS)
