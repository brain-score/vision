from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['r3m_resnet50_nocrop'] = ModelCommitment(identifier='r3m_resnet50_nocrop', activations_model=get_model('r3m_resnet50_nocrop'), layers=get_layers('r3m_resnet50_nocrop'))
model_registry['r3m_resnet34_nocrop'] = ModelCommitment(identifier='r3m_resnet34_nocrop', activations_model=get_model('r3m_resnet34_nocrop'), layers=get_layers('r3m_resnet34_nocrop'))
model_registry['r3m_resnet18_nocrop'] = ModelCommitment(identifier='r3m_resnet18_nocrop', activations_model=get_model('r3m_resnet18_nocrop'), layers=get_layers('r3m_resnet18_nocrop'))
