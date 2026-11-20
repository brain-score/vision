from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, IDENTIFIER, LAYERS, REGION_LAYER_MAP

model_registry['bao2020_objectspace_50d'] = lambda: ModelCommitment(
    identifier=IDENTIFIER,
    activations_model=get_model(),
    layers=LAYERS,
    region_layer_map=REGION_LAYER_MAP)
