from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["r50_tvpt"] = lambda: ModelCommitment(
    identifier="r50_tvpt",
    activations_model=get_model("r50_tvpt"),
    layers=get_layers("r50_tvpt"),
)
