from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['dbp_resnet50_julios'] = lambda: ModelCommitment(identifier='dbp_resnet50_julios', activations_model=get_model('dbp_resnet50_julios'), layers=get_layers('dbp_resnet50_julios'))
