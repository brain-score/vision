from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['CORnet-S'] = lambda: ModelCommitment(identifier='CORnet-S',
                                                               activations_model=get_model(),
                                                               layers=get_layers())