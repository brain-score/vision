from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eBarlow_lmda_02_200_full"] = lambda: ModelCommitment(
    identifier="eBarlow_lmda_02_200_full",
    activations_model=get_model("eBarlow_lmda_02_200_full"),
    layers=get_layers("eBarlow_lmda_02_200_full"),
)
