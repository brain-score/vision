from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['fitvid_trained_on_physion'] = ModelCommitment(identifier='fitvid_trained_on_physion', activations_model=get_model('fitvid_trained_on_physion'), layers=get_layers('fitvid_trained_on_physion'))
model_registry['fitvid_random'] = ModelCommitment(identifier='fitvid_random', activations_model=get_model('fitvid_random'), layers=get_layers('fitvid_random'))
