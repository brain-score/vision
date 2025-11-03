from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['hmax_v3_adj2'] = lambda: ModelCommitment(identifier='hmax_v3_adj2',
                                                        activations_model=get_model('hmax_v3_adj2'),
                                                        layers=get_layers('hmax_v3_adj2'),
                                                        visual_degrees=8)
