from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS


model_registry['ReAlnet01']= lambda: ModelCommitment(
    identifier='ReAlnet01',
    activations_model=get_model('ReAlnet01'),
    layers=LAYERS)

model_registry['ReAlnet02']= lambda: ModelCommitment(
    identifier='ReAlnet02',
    activations_model=get_model('ReAlnet02'),
    layers=LAYERS)

model_registry['ReAlnet03']= lambda: ModelCommitment(
    identifier='ReAlnet03',
    activations_model=get_model('ReAlnet03'),   
    layers=LAYERS)

model_registry['ReAlnet04']= lambda: ModelCommitment(
    identifier='ReAlnet04',
    activations_model=get_model('ReAlnet04'),
    layers=LAYERS)

model_registry['ReAlnet05']= lambda: ModelCommitment(
    identifier='ReAlnet05',
    activations_model=get_model('ReAlnet05'),
    layers=LAYERS)

model_registry['ReAlnet06']= lambda: ModelCommitment(
    identifier='ReAlnet06',
    activations_model=get_model('ReAlnet06'),
    layers=LAYERS)

model_registry['ReAlnet07']= lambda: ModelCommitment(
    identifier='ReAlnet07',
    activations_model=get_model('ReAlnet07'),
    layers=LAYERS)  

model_registry['ReAlnet08']= lambda: ModelCommitment(
    identifier='ReAlnet08',
    activations_model=get_model('ReAlnet08'),
    layers=LAYERS)  

model_registry['ReAlnet09']= lambda: ModelCommitment(
    identifier='ReAlnet09',
    activations_model=get_model('ReAlnet09'),
    layers=LAYERS)      

model_registry['ReAlnet10']= lambda: ModelCommitment(
    identifier='ReAlnet10',
    activations_model=get_model('ReAlnet10'),
    layers=LAYERS)      