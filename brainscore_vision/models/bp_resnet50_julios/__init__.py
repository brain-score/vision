from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


model_registry['bp_resnet50_julios'] = lambda: ModelCommitment(identifier='bp_resnet50_julios', activations_model=get_model('bp_resnet50_julios'), layers=get_layers('bp_resnet50_julios'))
