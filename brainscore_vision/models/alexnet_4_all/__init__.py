
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS

model_registry['alexnet_less_variation_iteration=3'] = lambda: ModelCommitment(identifier='alexnet_less_variation_iteration=3', activations_model=get_model('alexnet_less_variation_iteration=3'), layers=MODEL_CONFIGS['alexnet_less_variation_iteration=3']['model_commitment']['layers'])

model_registry['alexnet_less_variation_iteration=5'] = lambda: ModelCommitment(identifier='alexnet_less_variation_iteration=5', activations_model=get_model('alexnet_less_variation_iteration=5'), layers=MODEL_CONFIGS['alexnet_less_variation_iteration=5']['model_commitment']['layers'])

model_registry['alexnet_no_variation_iteration=1'] = lambda: ModelCommitment(identifier='alexnet_no_variation_iteration=1', activations_model=get_model('alexnet_no_variation_iteration=1'), layers=MODEL_CONFIGS['alexnet_no_variation_iteration=1']['model_commitment']['layers'])

model_registry['alexnet_no_variation_iteration=2'] = lambda: ModelCommitment(identifier='alexnet_no_variation_iteration=2', activations_model=get_model('alexnet_no_variation_iteration=2'), layers=MODEL_CONFIGS['alexnet_no_variation_iteration=2']['model_commitment']['layers'])

model_registry['alexnet_no_variation_iteration=3'] = lambda: ModelCommitment(identifier='alexnet_no_variation_iteration=3', activations_model=get_model('alexnet_no_variation_iteration=3'), layers=MODEL_CONFIGS['alexnet_no_variation_iteration=3']['model_commitment']['layers'])

model_registry['alexnet_no_variation_iteration=5'] = lambda: ModelCommitment(identifier='alexnet_no_variation_iteration=5', activations_model=get_model('alexnet_no_variation_iteration=5'), layers=MODEL_CONFIGS['alexnet_no_variation_iteration=5']['model_commitment']['layers'])

model_registry['alexnet_z_axis_iteration=5'] = lambda: ModelCommitment(identifier='alexnet_z_axis_iteration=5', activations_model=get_model('alexnet_z_axis_iteration=5'), layers=MODEL_CONFIGS['alexnet_z_axis_iteration=5']['model_commitment']['layers'])

model_registry['alexnet_wo_shading_iteration=1'] = lambda: ModelCommitment(identifier='alexnet_wo_shading_iteration=1', activations_model=get_model('alexnet_wo_shading_iteration=1'), layers=MODEL_CONFIGS['alexnet_wo_shading_iteration=1']['model_commitment']['layers'])


