from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['untrained'] = ModelCommitment(identifier='untrained', activations_model=get_model('untrained'), layers=get_layers('untrained'))
model_registry['supervised_e2e'] = ModelCommitment(identifier='supervised_e2e', activations_model=get_model('supervised_e2e'), layers=get_layers('supervised_e2e'))
model_registry['lpl'] = ModelCommitment(identifier='lpl', activations_model=get_model('lpl'), layers=get_layers('lpl'))
