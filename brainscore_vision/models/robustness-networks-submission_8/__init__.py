from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['alexnet_early_checkpoint'] = ModelCommitment(identifier='alexnet_early_checkpoint', activations_model=get_model('alexnet_early_checkpoint'), layers=get_layers('alexnet_early_checkpoint'))
model_registry['alexnet_reduced_aliasing_early_checkpoint'] = ModelCommitment(identifier='alexnet_reduced_aliasing_early_checkpoint', activations_model=get_model('alexnet_reduced_aliasing_early_checkpoint'), layers=get_layers('alexnet_reduced_aliasing_early_checkpoint'))
model_registry['vonealexnet_gaussian_noise_std4_fixed'] = ModelCommitment(identifier='vonealexnet_gaussian_noise_std4_fixed', activations_model=get_model('vonealexnet_gaussian_noise_std4_fixed'), layers=get_layers('vonealexnet_gaussian_noise_std4_fixed'))
