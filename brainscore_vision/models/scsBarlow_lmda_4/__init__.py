from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["scsBarlow_lmda_4"] = lambda: ModelCommitment(
    identifier="scsBarlow_lmda_4",
    activations_model=get_model("scsBarlow_lmda_4"),
    layers=get_layers("scsBarlow_lmda_4"),
)
