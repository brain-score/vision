from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['Hojin_Jang_ResNet50_Strong_blur'] = lambda: ModelCommitment(identifier='Hojin_Jang_ResNet50_Strong_blur', activations_model=get_model('Hojin_Jang_ResNet50_Strong_blur'), layers=get_layers('Hojin_Jang_ResNet50_Strong_blur'))
