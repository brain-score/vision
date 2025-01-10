from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import model_registry
from .model import get_layers,get_model


model_registry['resnet50_robust_l2_eps1'] = lambda: ModelCommitment(identifier='resnet50_robust_l2_eps1',
                                                                    activations_model=get_model('resnet50_robust_l2_eps1'),
                                                                    layers=get_layers('resnet50_robust_l2_eps1'))

