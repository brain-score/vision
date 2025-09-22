from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["mvimgnet_rf"] = lambda: ModelCommitment(
    identifier="mvimgnet_rf",
    activations_model=get_model("mvimgnet_rf"),
    layers=get_layers("mvimgnet_rf"),
)
