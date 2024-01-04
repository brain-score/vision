from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['omnivore_swinT'] = ModelCommitment(identifier='omnivore_swinT', activations_model=get_model('omnivore_swinT'), layers=get_layers('omnivore_swinT'))
model_registry['omnivore_swinS'] = ModelCommitment(identifier='omnivore_swinS', activations_model=get_model('omnivore_swinS'), layers=get_layers('omnivore_swinS'))
model_registry['omnivore_swinB'] = ModelCommitment(identifier='omnivore_swinB', activations_model=get_model('omnivore_swinB'), layers=get_layers('omnivore_swinB'))
model_registry['omnivore_swinB_imagenet21k'] = ModelCommitment(identifier='omnivore_swinB_imagenet21k', activations_model=get_model('omnivore_swinB_imagenet21k'), layers=get_layers('omnivore_swinB_imagenet21k'))
model_registry['omnivore_swinL_imagenet21k'] = ModelCommitment(identifier='omnivore_swinL_imagenet21k', activations_model=get_model('omnivore_swinL_imagenet21k'), layers=get_layers('omnivore_swinL_imagenet21k'))
