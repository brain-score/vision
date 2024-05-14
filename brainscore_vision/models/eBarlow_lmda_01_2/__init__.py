from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eBarlow_lmda_01_2"] = lambda: ModelCommitment(
    identifier="eBarlow_lmda_01_2",
    activations_model=get_model("eBarlow_lmda_01_2"),
    layers=get_layers("eBarlow_lmda_01_2"),
)
