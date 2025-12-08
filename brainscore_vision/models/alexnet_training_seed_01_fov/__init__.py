from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

### AlexNet_training_seed_01 with different fields of view than 8 visual degrees (default) ###
# exploring 4, 12, 16 visual degrees
# code is adapted from the original submission, please credit the original authors when using these models:
# https://github.com/brain-score/vision/tree/master/brainscore_vision/models/alexnet_training_seed_01
# models introduced in: Mehrer et al. (2020) "Individual differences among deep neural network models", URL: osf.io/3xupm

model_registry['alexnet_training_seed_01_fov4'] = lambda: ModelCommitment(identifier='alexnet_training_seed_01_fov4',
                                                                        activations_model=get_model('alexnet_training_seed_01_fov4'), 
                                                                        layers=get_layers('alexnet_training_seed_01_fov4'),
                                                                        visual_degrees=4)

model_registry['alexnet_training_seed_01_fov12'] = lambda: ModelCommitment(identifier='alexnet_training_seed_01_fov12',
                                                                         activations_model=get_model('alexnet_training_seed_01_fov12'), 
                                                                         layers=get_layers('alexnet_training_seed_01_fov12'),
                                                                         visual_degrees=12)

model_registry['alexnet_training_seed_01_fov16'] = lambda: ModelCommitment(identifier='alexnet_training_seed_01_fov16',
                                                                         activations_model=get_model('alexnet_training_seed_01_fov16'), 
                                                                         layers=get_layers('alexnet_training_seed_01_fov16'),
                                                                         visual_degrees=16)
    