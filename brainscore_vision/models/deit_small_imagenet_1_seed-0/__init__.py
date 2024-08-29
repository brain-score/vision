from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_COMMITMENT

MODEL_NAME = "deit_small"
MODEL_ID = "deit_small_imagenet_1_seed-0"

model_registry["deit_small_imagenet_1_seed-0"] = lambda: ModelCommitment(
    identifier=MODEL_ID,
    activations_model=get_model(),
    layers=MODEL_COMMITMENT["layers"],
    behavioral_readout_layer=MODEL_COMMITMENT["behavioral_readout_layer"],
    region_layer_map=MODEL_COMMITMENT["region_layer_map"],
)
