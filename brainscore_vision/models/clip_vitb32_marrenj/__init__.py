from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS, BEHAVIORAL_READOUT_LAYER

# No region_layer_map — brain-score's search picks the best visual.transformer
# block per region. behavioral_readout_layer is pinned to ln_post (the
# pre-projection 768-d CLS feature, matching our own alignment metric).
model_registry['clip_vitb32_marrenj'] = lambda: ModelCommitment(
    identifier='clip_vitb32_marrenj',
    activations_model=get_model(),
    layers=LAYERS,
    behavioral_readout_layer=BEHAVIORAL_READOUT_LAYER,
)
