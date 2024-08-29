from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_COMMITMENT

MODEL_NAME = "resnet34"
MODEL_ID = "resnet34_ecoset_full"

model_registry["resnet34_ecoset_full"] = lambda: ModelCommitment(
    identifier=MODEL_ID,
    activations_model=get_model(),
    layers=MODEL_COMMITMENT["layers"],
    behavioral_readout_layer=MODEL_COMMITMENT["behavioral_readout_layer"],
    region_layer_map=MODEL_COMMITMENT["region_layer_map"],
)
