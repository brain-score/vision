from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['omnivore_swinS'] = \
    lambda: ModelCommitment(identifier='omnivore_swinS', activations_model=get_model('omnivore_swinS'), layers=get_layers('omnivore_swinS'))