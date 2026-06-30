from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['Zejin_Lu_dvd_s_imagenet_resnet50'] = lambda: ModelCommitment(identifier='Zejin_Lu_dvd_s_imagenet_resnet50', activations_model=get_model('Zejin_Lu_dvd_s_imagenet_resnet50'), layers=get_layers('Zejin_Lu_dvd_s_imagenet_resnet50'))
