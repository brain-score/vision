from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['shufflenet_v2_neural_penalty2'] = lambda: ModelCommitment(identifier='shufflenet_v2_neural_penalty2', activations_model=get_model('shufflenet_v2_neural_penalty2'), layers=get_layers('shufflenet_v2_neural_penalty2'))
