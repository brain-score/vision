from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['EVA-02'] = lambda: ModelCommitment(identifier='EVA-02', activations_model=get_model('EVA-02'), layers=get_layers('EVA-02'))
