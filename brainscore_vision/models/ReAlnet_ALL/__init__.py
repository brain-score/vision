from brainscore_vision import model_registry
from .helpers.helpers import CORnetCommitment, _build_time_mappings
from .model import get_model, TIME_MAPPINGS, get_layers


model_registry['ReAlnet01_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet01_ALL', activations_model=get_model('ReAlnet01_ALL'),
                                                              layers=get_layers('ReAlnet01_ALL'),
                                                      time_mapping=_build_time_mappings(TIME_MAPPINGS))


model_registry['ReAlnet02_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet02_ALL', activations_model=get_model('ReAlnet02_ALL'),
                                                              layers=get_layers('ReAlnet02_ALL'),
                                                        time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet03_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet03_ALL', activations_model=get_model('ReAlnet03_ALL'),
                                                              layers = get_layers('ReAlnet03_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet04_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet04_ALL', activations_model=get_model('ReAlnet04_ALL'),
                                                              layers = get_layers('ReAlnet04_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet05_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet05_ALL', activations_model=get_model('ReAlnet05_ALL'),
                                                              layers = get_layers('ReAlnet05_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet06_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet06_ALL', activations_model=get_model('ReAlnet06_ALL'),
                                                              layers = get_layers('ReAlnet06_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet07_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet07_ALL', activations_model=get_model('ReAlnet07_ALL'),
                                                              layers = get_layers('ReAlnet07_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet08_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet08_ALL', activations_model=get_model('ReAlnet08_ALL'),
                                                              layers = get_layers('ReAlnet08_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet09_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet09_ALL', activations_model=get_model('ReAlnet09_ALL'),
                                                              layers = get_layers('ReAlnet09_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet10_ALL'] = lambda: CORnetCommitment(identifier='ReAlnet10_ALL', activations_model=get_model('ReAlnet10_ALL'),
                                                              layers = get_layers('ReAlnet10_ALL'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

