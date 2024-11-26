
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS
from .model import BLT_VS



model_registry['blt_vs'] = lambda: ModelCommitment(
    identifier='blt_vs',
    activations_model=get_model(),
    layers=LAYERS)

