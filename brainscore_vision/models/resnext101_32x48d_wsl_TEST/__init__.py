from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnext101_32x48d_wsl_TEST'] = lambda: ModelCommitment(identifier='resnext101_32x48d_wsl_TEST',
                                                               activations_model=get_model('resnext101_32x48d_wsl_TEST'),
                                                               layers=get_layers('resnext101_32x48d_wsl_TEST'))