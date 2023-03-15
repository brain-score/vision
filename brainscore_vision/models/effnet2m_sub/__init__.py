from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['effnetv2m_custom384'] = ModelCommitment(identifier='effnetv2m_custom384', activations_model=get_model('effnetv2m_custom384'), layers=get_layers('effnetv2m_custom384'))
