from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['r3m_50'] = ModelCommitment(identifier='r3m_50', activations_model=get_model('r3m_50'), layers=get_layers('r3m_50'))
model_registry['r3m_34'] = ModelCommitment(identifier='r3m_34', activations_model=get_model('r3m_34'), layers=get_layers('r3m_34'))
model_registry['r3m_18'] = ModelCommitment(identifier='r3m_18', activations_model=get_model('r3m_18'), layers=get_layers('r3m_18'))
