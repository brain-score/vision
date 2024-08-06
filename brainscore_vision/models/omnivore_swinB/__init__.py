from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['omnivore_swinB'] = \
    lambda: ModelCommitment(identifier='omnivore_swinB', activations_model=get_model('omnivore_swinB'), layers=get_layers('omnivore_swinB'))