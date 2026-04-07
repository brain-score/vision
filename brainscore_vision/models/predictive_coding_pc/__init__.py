from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['predictive_coding_pc'] = lambda: ModelCommitment(
    identifier='predictive_coding_pc',
    activations_model=get_model('predictive_coding_pc'),
    layers=get_layers('predictive_coding_pc'),
    region_layer_map={'V1': 'r0', 'V2': 'r1', 'V4': 'r1', 'IT': 'r3'},
)
