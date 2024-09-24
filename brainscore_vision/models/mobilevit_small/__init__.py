from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['mobilevit_small'] = lambda: ModelCommitment(identifier='mobilevit_small',
                                                            activations_model=get_model('mobilevit_small'),
                                                            layers=get_layers('mobilevit_small'))