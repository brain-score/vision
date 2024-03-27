from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["eMMCR_lmda_01_1"] = lambda: ModelCommitment(
    identifier="eMMCR_lmda_01_1",
    activations_model=get_model("eMMCR_lmda_01_1"),
    layers=get_layers("eMMCR_lmda_01_1"),
)
