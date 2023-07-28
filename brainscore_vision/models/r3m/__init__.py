from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['r3m_resnet50'] = ModelCommitment(identifier='r3m_resnet50', activations_model=get_model('r3m_resnet50'), layers=get_layers('r3m_resnet50'))
model_registry['r3m_resnet34'] = ModelCommitment(identifier='r3m_resnet34', activations_model=get_model('r3m_resnet34'), layers=get_layers('r3m_resnet34'))
model_registry['r3m_resnet18'] = ModelCommitment(identifier='r3m_resnet18', activations_model=get_model('r3m_resnet18'), layers=get_layers('r3m_resnet18'))
