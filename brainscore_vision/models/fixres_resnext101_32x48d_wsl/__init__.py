from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['fixres_resnext101_32x48d_wsl'] = lambda: ModelCommitment(identifier='fixres_resnext101_32x48d_wsl',
                                                               activations_model=get_model(),
                                                               layers=get_layers('fixres_resnext101_32x48d_wsl'))