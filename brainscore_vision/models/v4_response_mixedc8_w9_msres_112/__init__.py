from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

from .model import IDENTIFIER, LAYERS, get_model


model_registry[IDENTIFIER] = lambda: ModelCommitment(
    identifier=IDENTIFIER,
    activations_model=get_model(),
    layers=LAYERS,
    behavioral_readout_layer="taps.head4",
    region_layer_map={
        "V1": "taps.core1",
        "V2": "taps.core2",
        "V4": "taps.core4",
        "IT": "taps.head3",
    },
    visual_degrees=8,
)
