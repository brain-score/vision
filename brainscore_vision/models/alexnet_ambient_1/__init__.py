
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS
model_registry['alexnet_ambient_iteration=1'] = lambda: ModelCommitment(identifier='alexnet_ambient_iteration=1', activations_model=get_model('alexnet_ambient_iteration=1'), layers=MODEL_CONFIGS['alexnet_ambient_iteration=1']['model_commitment']['layers'])
        