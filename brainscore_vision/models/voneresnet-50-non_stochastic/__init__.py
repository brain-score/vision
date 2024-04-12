from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['voneresnet-50-non_stochastic'] = lambda: ModelCommitment(identifier='voneresnet-50-non_stochastic',
                                                               activations_model=get_model(),
                                                               layers=get_layers())