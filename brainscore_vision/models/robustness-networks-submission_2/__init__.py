from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet50'] = ModelCommitment(identifier='resnet50', activations_model=get_model('resnet50'), layers=get_layers('resnet50'))
model_registry['resnet50_l2_3_robust'] = ModelCommitment(identifier='resnet50_l2_3_robust', activations_model=get_model('resnet50_l2_3_robust'), layers=get_layers('resnet50_l2_3_robust'))
model_registry['resnet50_linf_4_robust'] = ModelCommitment(identifier='resnet50_linf_4_robust', activations_model=get_model('resnet50_linf_4_robust'), layers=get_layers('resnet50_linf_4_robust'))
model_registry['resnet50_linf_8_robust'] = ModelCommitment(identifier='resnet50_linf_8_robust', activations_model=get_model('resnet50_linf_8_robust'), layers=get_layers('resnet50_linf_8_robust'))
model_registry['resnet50_random_l2_perturb'] = ModelCommitment(identifier='resnet50_random_l2_perturb', activations_model=get_model('resnet50_random_l2_perturb'), layers=get_layers('resnet50_random_l2_perturb'))
model_registry['resnet50_random_linf8_perturb'] = ModelCommitment(identifier='resnet50_random_linf8_perturb', activations_model=get_model('resnet50_random_linf8_perturb'), layers=get_layers('resnet50_random_linf8_perturb'))
