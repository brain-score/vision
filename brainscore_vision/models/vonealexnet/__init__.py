from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["vone_alexnet"] = lambda: ModelCommitment(
    identifier="vone_alexnet",
    activations_model=get_model("vone_alexnet"),
    layers=get_layers("vone_alexnet"),
)

model_registry["vone_alexnet_full"] = lambda: ModelCommitment(
    identifier="vone_alexnet_full",
    activations_model=get_model("vone_alexnet_full"),
    layers=get_layers("vone_alexnet_full"),
)
