from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['effnetb2_cutmixpatch_robust35_avgee5e3_manylayers_424x377'] = lambda: ModelCommitment(
    identifier='effnetb2_cutmixpatch_robust35_avgee5e3_manylayers_424x377',
    activations_model=get_model(),
    layers=LAYERS)
