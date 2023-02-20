from brainscore_vision import model_registry
from brainscore_vision.model_helpers import ModelCommitment
from .model import get_model, get_layers

model_registry['untrained_cifar'] = ModelCommitment(identifier='untrained_cifar', activations_model=get_model('untrained_cifar'), layers=get_layers('untrained_cifar'))
model_registry['supervised_e2e_cifar'] = ModelCommitment(identifier='supervised_e2e_cifar', activations_model=get_model('supervised_e2e_cifar'), layers=get_layers('supervised_e2e_cifar'))
model_registry['supervised_cifar'] = ModelCommitment(identifier='supervised_cifar', activations_model=get_model('supervised_cifar'), layers=get_layers('supervised_cifar'))
model_registry['lpl_e2e_cifar'] = ModelCommitment(identifier='lpl_e2e_cifar', activations_model=get_model('lpl_e2e_cifar'), layers=get_layers('lpl_e2e_cifar'))
model_registry['lpl_cifar'] = ModelCommitment(identifier='lpl_cifar', activations_model=get_model('lpl_cifar'), layers=get_layers('lpl_cifar'))
model_registry['neg_samples_e2e_cifar'] = ModelCommitment(identifier='neg_samples_e2e_cifar', activations_model=get_model('neg_samples_e2e_cifar'), layers=get_layers('neg_samples_e2e_cifar'))
model_registry['neg_samples_cifar'] = ModelCommitment(identifier='neg_samples_cifar', activations_model=get_model('neg_samples_cifar'), layers=get_layers('neg_samples_cifar'))
