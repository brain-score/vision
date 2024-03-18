from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['lagani-timagenet_fc_on_hebb_fc5'] = lambda: ModelCommitment(
    identifier='lagani-timagenet_fc_on_hebb_fc5',
    activations_model=get_model(),
    layers=LAYERS)
