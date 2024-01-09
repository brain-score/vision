from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['my-weights-model'] = lambda: ModelCommitment(identifier='my-weights-model', activations_model=get_model('my-weights-model'), layers=get_layers('my-weights-model'))
