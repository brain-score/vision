from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["CE-Barlow_lmda_04"] = lambda: ModelCommitment(
    identifier="CE-Barlow_lmda_04",
    activations_model=get_model("CE-Barlow_lmda_04"),
    layers=get_layers("CE-Barlow_lmda_04"),
)
