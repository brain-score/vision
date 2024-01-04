from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-50x2_untrained'] = ModelCommitment(identifier='resnet-50x2_untrained', activations_model=get_model('resnet-50x2_untrained'), layers=get_layers('resnet-50x2_untrained'))
model_registry['resnet-50x4_untrained'] = ModelCommitment(identifier='resnet-50x4_untrained', activations_model=get_model('resnet-50x4_untrained'), layers=get_layers('resnet-50x4_untrained'))
