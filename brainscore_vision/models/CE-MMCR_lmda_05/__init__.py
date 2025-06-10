from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["CE-MMCR_lmda_05"] = lambda: ModelCommitment(
    identifier="CE-MMCR_lmda_05",
    activations_model=get_model("CE-MMCR_lmda_05"),
    layers=get_layers("CE-MMCR_lmda_05"),
)
