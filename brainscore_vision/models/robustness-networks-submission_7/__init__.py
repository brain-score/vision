from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['voneresnet50_fixed_noise'] = ModelCommitment(identifier='voneresnet50_fixed_noise', activations_model=get_model('voneresnet50_fixed_noise'), layers=get_layers('voneresnet50_fixed_noise'))
model_registry['gvoneresnet50_fixed_noise'] = ModelCommitment(identifier='gvoneresnet50_fixed_noise', activations_model=get_model('gvoneresnet50_fixed_noise'), layers=get_layers('gvoneresnet50_fixed_noise'))
