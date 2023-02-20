from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['resnet-50-ANT3x3_SIN'] = ModelCommitment(identifier='resnet-50-ANT3x3_SIN', activations_model=get_model('resnet-50-ANT3x3_SIN'), layers=get_layers('resnet-50-ANT3x3_SIN'))
model_registry['resnet-34-pt'] = ModelCommitment(identifier='resnet-34-pt', activations_model=get_model('resnet-34-pt'), layers=get_layers('resnet-34-pt'))
model_registry['resnet-101-pt'] = ModelCommitment(identifier='resnet-101-pt', activations_model=get_model('resnet-101-pt'), layers=get_layers('resnet-101-pt'))
model_registry['resnet-152-pt'] = ModelCommitment(identifier='resnet-152-pt', activations_model=get_model('resnet-152-pt'), layers=get_layers('resnet-152-pt'))
model_registry['resnext-50-32x4d-pt'] = ModelCommitment(identifier='resnext-50-32x4d-pt', activations_model=get_model('resnext-50-32x4d-pt'), layers=get_layers('resnext-50-32x4d-pt'))
model_registry['resnext-101-32x8d-pt'] = ModelCommitment(identifier='resnext-101-32x8d-pt', activations_model=get_model('resnext-101-32x8d-pt'), layers=get_layers('resnext-101-32x8d-pt'))
model_registry['mnasnet0_5-pt'] = ModelCommitment(identifier='mnasnet0_5-pt', activations_model=get_model('mnasnet0_5-pt'), layers=get_layers('mnasnet0_5-pt'))
model_registry['mnasnet1_0-pt'] = ModelCommitment(identifier='mnasnet1_0-pt', activations_model=get_model('mnasnet1_0-pt'), layers=get_layers('mnasnet1_0-pt'))
model_registry['resnet-50-ANT3x3_SIN'] = ModelCommitment(identifier='resnet-50-ANT3x3_SIN', activations_model=get_model('resnet-50-ANT3x3_SIN'), layers=get_layers('resnet-50-ANT3x3_SIN'))
