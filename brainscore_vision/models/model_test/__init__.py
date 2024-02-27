# dummy change
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

model_registry['yudixie_resnet50_imagenet1kpret_0_240222'] = lambda: ModelCommitment(identifier='yudixie_resnet50_imagenet1kpret_0_240222',
                                                                                     activations_model=get_model('yudixie_resnet50_imagenet1kpret_0_240222'),
                                                                                     layers=get_layers('yudixie_resnet50_imagenet1kpret_0_240222'))
