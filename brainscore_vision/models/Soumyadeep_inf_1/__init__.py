from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['Soumyadeep_inf_1'] = lambda: ModelCommitment(identifier='Soumyadeep_inf_1', activations_model=get_model('Soumyadeep_inf_1'), layers=get_layers('Soumyadeep_inf_1'))
