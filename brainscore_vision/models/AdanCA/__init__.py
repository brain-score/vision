from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers

# model_registry['my_custom_model'] = lambda: ModelCommitment(identifier='my_custom_model', activations_model=get_model('my_custom_model'), layers=get_layers('my_custom_model'))



for name in ["swin_base_patch4_window7_224", "convit_base", "rvt_base_plus", "fan_base_hybrid",
            "swin_base_nca_version_self_trained_imagenet", "convit_base_nca_version_self_trained_imagenet", "rvt_base_plus_nca_version_self_trained_imagenet", "fan_base_hybrid_nca_version_self_trained_imagenet"]:
    model_registry[name] = lambda name=name: ModelCommitment(identifier=name, activations_model=get_model(name), layers=get_layers(name))
