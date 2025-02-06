from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['BiT-S-R50x3'] = \
    lambda: ModelCommitment(identifier='BiT-S-R50x3', activations_model=get_model('BiT-S-R50x3'), layers=get_layers('BiT-S-R50x3'))