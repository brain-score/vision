from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet18_barlortwins:standard_in1k_ba1024_ep100'] = lambda: ModelCommitment(identifier='resnet18_barlortwins:standard_in1k_ba1024_ep100', activations_model=get_model('resnet18_barlortwins:standard_in1k_ba1024_ep100'), layers=get_layers('resnet18_barlortwins:standard_in1k_ba1024_ep100'))
