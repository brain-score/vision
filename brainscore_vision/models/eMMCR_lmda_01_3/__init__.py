from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eMMCR_lmda_01_3"] = lambda: ModelCommitment(
    identifier="eMMCR_lmda_01_3",
    activations_model=get_model("eMMCR_lmda_01_3"),
    layers=get_layers("eMMCR_lmda_01_3"),
)
