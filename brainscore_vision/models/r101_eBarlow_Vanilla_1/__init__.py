from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["r101_eBarlow_Vanilla_1"] = lambda: ModelCommitment(
    identifier="r101_eBarlow_Vanilla_1",
    activations_model=get_model("r101_eBarlow_Vanilla_1"),
    layers=get_layers("r101_eBarlow_Vanilla_1"),
)
