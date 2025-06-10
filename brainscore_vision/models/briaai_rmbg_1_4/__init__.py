from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['briaai_rmbg_1_4'] = lambda: ModelCommitment(identifier='briaai_rmbg_1_4',
                                                               activations_model=get_model('briaai_rmbg_1_4'),
                                                               layers=get_layers('briaai_rmbg_1_4'))