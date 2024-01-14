from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry["vonegrcnn_52e_full"] = lambda: ModelCommitment(
    identifier="vonegrcnn_52e_full",
    activations_model=get_model(),
    layers=LAYERS,
)
