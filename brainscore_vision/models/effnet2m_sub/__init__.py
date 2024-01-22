from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

model_registry['effnetv2m_custom384'] = lambda: ModelCommitment(
    identifier='effnetv2m_custom384',
    activations_model=get_model(),
    layers=LAYERS)
