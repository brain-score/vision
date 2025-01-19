from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers


def commit_model(identifier):
    return ModelCommitment(identifier=identifier,
                           activations_model=get_model(identifier),
                           layers=get_layers(identifier))

model_registry['yudixie_resnet50_translation_rotation_0_240908'] = lambda: commit_model('yudixie_resnet50_translation_rotation_0_240908')
