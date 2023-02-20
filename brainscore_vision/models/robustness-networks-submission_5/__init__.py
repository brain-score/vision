from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50_simclr'] = ModelCommitment(identifier='resnet50_simclr', activations_model=get_model('resnet50_simclr'), layers=get_layers('resnet50_simclr'))
model_registry['resnet50_moco_v2'] = ModelCommitment(identifier='resnet50_moco_v2', activations_model=get_model('resnet50_moco_v2'), layers=get_layers('resnet50_moco_v2'))
