from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet'] = ModelCommitment(identifier='alexnet', activations_model=get_model('alexnet'), layers=get_layers('alexnet'))
model_registry['alexnet_l2_3_robust'] = ModelCommitment(identifier='alexnet_l2_3_robust', activations_model=get_model('alexnet_l2_3_robust'), layers=get_layers('alexnet_l2_3_robust'))
model_registry['alexnet_random_l2_3_perturb'] = ModelCommitment(identifier='alexnet_random_l2_3_perturb', activations_model=get_model('alexnet_random_l2_3_perturb'), layers=get_layers('alexnet_random_l2_3_perturb'))
model_registry['alexnet_linf_8_robust'] = ModelCommitment(identifier='alexnet_linf_8_robust', activations_model=get_model('alexnet_linf_8_robust'), layers=get_layers('alexnet_linf_8_robust'))
model_registry['alexnet_random_linf8_perturb'] = ModelCommitment(identifier='alexnet_random_linf8_perturb', activations_model=get_model('alexnet_random_linf8_perturb'), layers=get_layers('alexnet_random_linf8_perturb'))
