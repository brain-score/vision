from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['cb-alexnet-tutorial-model'] = ModelCommitment(identifier='cb-alexnet-tutorial-model', activations_model=get_model('cb-alexnet-tutorial-model'), layers=get_layers('cb-alexnet-tutorial-model'))
