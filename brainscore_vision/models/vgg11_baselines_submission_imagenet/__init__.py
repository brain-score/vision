from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['untrained_imagenet'] = ModelCommitment(identifier='untrained_imagenet', activations_model=get_model('untrained_imagenet'), layers=get_layers('untrained_imagenet'))
model_registry['supervised_e2e_imagenet'] = ModelCommitment(identifier='supervised_e2e_imagenet', activations_model=get_model('supervised_e2e_imagenet'), layers=get_layers('supervised_e2e_imagenet'))
model_registry['supervised_imagenet'] = ModelCommitment(identifier='supervised_imagenet', activations_model=get_model('supervised_imagenet'), layers=get_layers('supervised_imagenet'))
model_registry['lpl_e2e_imagenet'] = ModelCommitment(identifier='lpl_e2e_imagenet', activations_model=get_model('lpl_e2e_imagenet'), layers=get_layers('lpl_e2e_imagenet'))
model_registry['lpl_imagenet'] = ModelCommitment(identifier='lpl_imagenet', activations_model=get_model('lpl_imagenet'), layers=get_layers('lpl_imagenet'))
