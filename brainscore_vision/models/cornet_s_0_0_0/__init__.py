from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

IDENTIFIER = 'cornet_s_0_0_0'

model_registry[IDENTIFIER] = lambda: ModelCommitment(identifier=IDENTIFIER, activations_model=get_model(IDENTIFIER), layers=get_layers(IDENTIFIER))
