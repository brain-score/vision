from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_layers, get_model

# Trained Mehrer et al. (2020) AlexNets from osf.io/3xupm; the untrained alexnet_training_seed_* plugins stay as baselines

model_registry['mehrer2020_alexnet_training_seed_01'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_01', activations_model=get_model('mehrer2020_alexnet_training_seed_01'), layers=get_layers('mehrer2020_alexnet_training_seed_01'))
model_registry['mehrer2020_alexnet_training_seed_02'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_02', activations_model=get_model('mehrer2020_alexnet_training_seed_02'), layers=get_layers('mehrer2020_alexnet_training_seed_02'))
model_registry['mehrer2020_alexnet_training_seed_03'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_03', activations_model=get_model('mehrer2020_alexnet_training_seed_03'), layers=get_layers('mehrer2020_alexnet_training_seed_03'))
model_registry['mehrer2020_alexnet_training_seed_04'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_04', activations_model=get_model('mehrer2020_alexnet_training_seed_04'), layers=get_layers('mehrer2020_alexnet_training_seed_04'))
model_registry['mehrer2020_alexnet_training_seed_05'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_05', activations_model=get_model('mehrer2020_alexnet_training_seed_05'), layers=get_layers('mehrer2020_alexnet_training_seed_05'))
model_registry['mehrer2020_alexnet_training_seed_06'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_06', activations_model=get_model('mehrer2020_alexnet_training_seed_06'), layers=get_layers('mehrer2020_alexnet_training_seed_06'))
model_registry['mehrer2020_alexnet_training_seed_07'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_07', activations_model=get_model('mehrer2020_alexnet_training_seed_07'), layers=get_layers('mehrer2020_alexnet_training_seed_07'))
model_registry['mehrer2020_alexnet_training_seed_08'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_08', activations_model=get_model('mehrer2020_alexnet_training_seed_08'), layers=get_layers('mehrer2020_alexnet_training_seed_08'))
model_registry['mehrer2020_alexnet_training_seed_09'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_09', activations_model=get_model('mehrer2020_alexnet_training_seed_09'), layers=get_layers('mehrer2020_alexnet_training_seed_09'))
model_registry['mehrer2020_alexnet_training_seed_10'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_10', activations_model=get_model('mehrer2020_alexnet_training_seed_10'), layers=get_layers('mehrer2020_alexnet_training_seed_10'))
model_registry['mehrer2020_alexnet_training_seed_01_fov4'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_01_fov4', activations_model=get_model('mehrer2020_alexnet_training_seed_01_fov4'), layers=get_layers('mehrer2020_alexnet_training_seed_01_fov4'), visual_degrees=4)
model_registry['mehrer2020_alexnet_training_seed_01_fov12'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_01_fov12', activations_model=get_model('mehrer2020_alexnet_training_seed_01_fov12'), layers=get_layers('mehrer2020_alexnet_training_seed_01_fov12'), visual_degrees=12)
model_registry['mehrer2020_alexnet_training_seed_01_fov16'] = lambda: ModelCommitment(identifier='mehrer2020_alexnet_training_seed_01_fov16', activations_model=get_model('mehrer2020_alexnet_training_seed_01_fov16'), layers=get_layers('mehrer2020_alexnet_training_seed_01_fov16'), visual_degrees=16)
