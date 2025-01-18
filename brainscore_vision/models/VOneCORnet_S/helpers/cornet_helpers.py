import logging
from importlib import import_module
from brainscore_vision.model_helpers.activations import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from .cornet_s_helpers import cornet as cornet_ctr
from vonenet import get_model
from .cornet_s_helpers import TemporalPytorchWrapper
import functools


_logger = logging.getLogger(__name__)


def torchvision_model(identifier, image_size):
    module = import_module(f'torchvision.models')
    model_ctr = getattr(module, identifier)
    from model_helpers.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=identifier, model=model_ctr(pretrained=True), preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def cornet(*args, **kwargs):  # wrapper to avoid having to import cornet at top-level
    return cornet_ctr(*args, **kwargs)

def vonecornet(model_name='cornets'):
    model = get_model(model_name)
    model = model.module
    preprocessing = functools.partial(load_preprocess_images, image_size=224,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = TemporalPytorchWrapper(identifier='vone'+model_name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

