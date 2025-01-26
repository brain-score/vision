
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers
model_registry['clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5'] = lambda: ModelCommitment(identifier='resnet50', activations_model=get_model('clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5'), layers=get_layers('clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5')))
