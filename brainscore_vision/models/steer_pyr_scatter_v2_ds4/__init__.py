from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry["steer_pyr_scatter_v2_ds4"] = lambda: ModelCommitment(
    identifier="steer_pyr_scatter_v2_ds4",
    activations_model=get_model("steer_pyr_scatter_v2_ds4"),
    layers=get_layers("steer_pyr_scatter_v2_ds4"),
)
