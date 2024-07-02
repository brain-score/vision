from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eBarlow_augself_mlp_1"] = lambda: ModelCommitment(
    identifier="eBarlow_augself_mlp_1",
    activations_model=get_model("eBarlow_augself_mlp_1"),
    layers=get_layers("eBarlow_augself_mlp_1"),
)
