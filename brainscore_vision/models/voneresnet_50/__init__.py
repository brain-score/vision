from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['voneresnet-50'] = \
    lambda: ModelCommitment(identifier='voneresnet-50', activations_model=get_model('voneresnet-50'), layers=get_layers('voneresnet-50'))