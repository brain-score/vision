from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["sBarlow_lmda_1"] = lambda: ModelCommitment(
    identifier="sBarlow_lmda_1",
    activations_model=get_model("sBarlow_lmda_1"),
    layers=get_layers("sBarlow_lmda_1"),
)
