from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["r101_eBarlow_lmda_02_1_copy"] = lambda: ModelCommitment(
    identifier="r101_eBarlow_lmda_02_1_copy",
    activations_model=get_model("r101_eBarlow_lmda_02_1_copy"),
    layers=get_layers("r101_eBarlow_lmda_02_1_copy"),
)
