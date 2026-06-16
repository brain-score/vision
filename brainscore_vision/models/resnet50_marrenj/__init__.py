from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS, REGION_LAYER_MAP, BEHAVIORAL_READOUT_LAYER

model_registry['resnet50_marrenj'] = lambda: ModelCommitment(
    identifier='resnet50_marrenj',
    activations_model=get_model(),
    layers=LAYERS,
    region_layer_map=REGION_LAYER_MAP,
    behavioral_readout_layer=BEHAVIORAL_READOUT_LAYER,
)
