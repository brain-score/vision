from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# region_layer_map intentionally omitted -> framework auto-selects the best candidate
# layer per region (V1/V2/V4/IT) via LayerSelection on the public neural benchmarks.
# behavioral_readout_layer pinned to the high-level pooled feature.
model_registry['convnext_large_clip_laion2b_finelayers'] = lambda: ModelCommitment(
    identifier='convnext_large_clip_laion2b_finelayers',
    activations_model=get_model('convnext_large_clip_laion2b_finelayers'),
    layers=get_layers('convnext_large_clip_laion2b_finelayers'),
    behavioral_readout_layer='head.global_pool',
    visual_degrees=8,
)
