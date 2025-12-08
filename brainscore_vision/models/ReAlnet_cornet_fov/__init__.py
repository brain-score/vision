from brainscore_vision import model_registry
from .helpers.helpers import CORnetCommitment, _build_time_mappings
from .model import get_model, TIME_MAPPINGS, get_layers

### ReAlnet01_cornet with different fields of view than 8 visual degrees (default) ###
# exploring 4, 12, 16 visual degrees
# code is adapted from the original submission, please credit the original authors when using these models:
# https://github.com/brain-score/vision/tree/master/brainscore_vision/models/ReAlnet_cornet

model_registry['ReAlnet01_cornet_fov4'] = lambda: CORnetCommitment(identifier='ReAlnet01_cornet_fov4', activations_model=get_model('ReAlnet01_cornet_fov4'),
                                                                  layers=get_layers('ReAlnet01_cornet_fov4'),
                                                                  time_mapping=_build_time_mappings(TIME_MAPPINGS),
                                                                  visual_degrees=4)

model_registry['ReAlnet01_cornet_fov12'] = lambda: CORnetCommitment(identifier='ReAlnet01_cornet_fov12', activations_model=get_model('ReAlnet01_cornet_fov12'),
                                                                  layers=get_layers('ReAlnet01_cornet_fov12'),
                                                                  time_mapping=_build_time_mappings(TIME_MAPPINGS),
                                                                  visual_degrees=12)
model_registry['ReAlnet01_cornet_fov16'] = lambda: CORnetCommitment(identifier='ReAlnet01_cornet_fov16', activations_model=get_model('ReAlnet01_cornet_fov16'),
                                                                  layers=get_layers('ReAlnet01_cornet_fov16'),
                                                                  time_mapping=_build_time_mappings(TIME_MAPPINGS),
                                                                  visual_degrees=16)
