
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS, get_layers

model_registry['alexnet_less_variation_iteration=3'] = lambda: ModelCommitment(identifier='alexnet_less_variation_iteration=3', activations_model=get_model('alexnet_less_variation_iteration=3'), layers=MODEL_CONFIGS['alexnet_less_variation_iteration=3']['model_commitment']['layers'])


