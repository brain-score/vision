from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_training_seed_06'] = lambda: ModelCommitment(identifier='alexnet_training_seed_06', activations_model=get_model('alexnet_training_seed_06'), layers=get_layers('alexnet_training_seed_06'))
    