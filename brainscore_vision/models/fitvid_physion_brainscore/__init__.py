from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['fitvid_trained_on_physion_alllayers'] = ModelCommitment(identifier='fitvid_trained_on_physion_alllayers', activations_model=get_model('fitvid_trained_on_physion_alllayers'), layers=get_layers('fitvid_trained_on_physion_alllayers'))
model_registry['fitvid_random_alllayers'] = ModelCommitment(identifier='fitvid_random_alllayers', activations_model=get_model('fitvid_random_alllayers'), layers=get_layers('fitvid_random_alllayers'))
