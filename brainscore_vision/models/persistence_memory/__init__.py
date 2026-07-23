from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_layers, get_model


model_registry["persistence_memory"] = lambda: ModelCommitment(
    identifier="persistence_memory",
    activations_model=get_model("persistence_memory"),
    layers=get_layers("persistence_memory"),
)
