# __init__.py - 모델 등록
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# hk_model_1 등록
model_registry['hk_model_1'] = lambda: ModelCommitment(
    identifier='hk_model_1',
    activations_model=get_model('hk_model_1'),
    layers=get_layers('hk_model_1')
)