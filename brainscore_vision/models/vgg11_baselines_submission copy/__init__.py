from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['supervised'] = ModelCommitment(identifier='supervised', activations_model=get_model('supervised'), layers=get_layers('supervised'))
model_registry['lpl_e2e'] = ModelCommitment(identifier='lpl_e2e', activations_model=get_model('lpl_e2e'), layers=get_layers('lpl_e2e'))
model_registry['neg_samples'] = ModelCommitment(identifier='neg_samples', activations_model=get_model('neg_samples'), layers=get_layers('neg_samples'))
