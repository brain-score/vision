from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['fast_2px_step2_eps2_repeat1_trial1_model_best'] = lambda: ModelCommitment(identifier='fast_2px_step2_eps2_repeat1_trial1_model_best', activations_model=get_model('fast_2px_step2_eps2_repeat1_trial1_model_best'), layers=get_layers('fast_2px_step2_eps2_repeat1_trial1_model_best'))
