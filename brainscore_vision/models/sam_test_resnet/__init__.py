from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['sam_test_resnet'] = lambda: ModelCommitment(identifier='sam_test_resnet', activations_model=get_model('sam_test_resnet'), layers=get_layers('sam_test_resnet'))
