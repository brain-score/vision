from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["CE-SimCLR_lmda_0001"] = lambda: ModelCommitment(
    identifier="CE-SimCLR_lmda_0001",
    activations_model=get_model("CE-SimCLR_lmda_0001"),
    layers=get_layers("CE-SimCLR_lmda_0001"),
)
