from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['shufflenet_neuralpca_penalty'] = lambda: ModelCommitment(identifier='shufflenet_neuralpca_penalty', activations_model=get_model('shufflenet_neuralpca_penalty'), layers=get_layers('shufflenet_neuralpca_penalty'))
