from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_COMMITMENT

MODEL_NAME = "efficientnet_b0"
MODEL_ID = "efficientnet_b0_imagenet_full"

model_registry["efficientnet_b0_imagenet_full"] = lambda: ModelCommitment(
    identifier=MODEL_ID,
    activations_model=get_model(),
    layers=MODEL_COMMITMENT["layers"],
    behavioral_readout_layer=MODEL_COMMITMENT["behavioral_readout_layer"],
    region_layer_map=MODEL_COMMITMENT["region_layer_map"],
)
