# Created by David Coggan on 2025 03 13
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['regnet_y_128gf_E2E'] = lambda: ModelCommitment(
    identifier='regnet_y_128gf_E2E', activations_model=get_model(
        'regnet_y_128gf_E2E'), layers=get_layers('regnet_y_128gf_E2E'))
