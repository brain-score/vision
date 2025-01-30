from brainscore_vision import model_registry
from .helpers.helpers import CORnetCommitment, _build_time_mappings
from .model import get_model, TIME_MAPPINGS, get_layers


model_registry['ReAlnet01_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet01_cornet', activations_model=get_model('ReAlnet01_cornet'),
                                                              layers=get_layers('ReAlnet01_cornet'),
                                                      time_mapping=_build_time_mappings(TIME_MAPPINGS))


model_registry['ReAlnet02_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet02_cornet', activations_model=get_model('ReAlnet02_cornet'),
                                                              layers=get_layers('ReAlnet02_cornet'),
                                                        time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet03_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet03_cornet', activations_model=get_model('ReAlnet03_cornet'),
                                                              layers = get_layers('ReAlnet03_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet04_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet04_cornet', activations_model=get_model('ReAlnet04_cornet'),
                                                              layers = get_layers('ReAlnet04_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet05_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet05_cornet', activations_model=get_model('ReAlnet05_cornet'),
                                                              layers = get_layers('ReAlnet05_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet06_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet06_cornet', activations_model=get_model('ReAlnet06_cornet'),
                                                              layers = get_layers('ReAlnet06_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet07_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet07_cornet', activations_model=get_model('ReAlnet07_cornet'),
                                                              layers = get_layers('ReAlnet07_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet08_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet08_cornet', activations_model=get_model('ReAlnet08_cornet'),
                                                              layers = get_layers('ReAlnet08_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet09_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet09_cornet', activations_model=get_model('ReAlnet09_cornet'),
                                                              layers = get_layers('ReAlnet09_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

model_registry['ReAlnet10_cornet'] = lambda: CORnetCommitment(identifier='ReAlnet10_cornet', activations_model=get_model('ReAlnet10_cornet'),
                                                              layers = get_layers('ReAlnet10_cornet'),
                                                                time_mapping=_build_time_mappings(TIME_MAPPINGS))

