from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["vonegrcnn_47e"] = lambda: ModelCommitment(
    identifier="vonegrcnn_47e",
    activations_model=get_model(),
    layers=LAYERS,
)
