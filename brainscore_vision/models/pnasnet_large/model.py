from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import logging

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)

'''
This is a Pytorch implementation of pnasnet_large.

Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://huggingface.co/timm/pnasnet5large.tf_in1k

Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow
implementation.

'''

_cached_model = None


def _load_model(use_half_precision: bool = True):
    """Load model with optional half precision for reduced memory usage."""
    import torch
    import timm

    global _cached_model
    if _cached_model is not None:
        return _cached_model

    model = timm.create_model('pnasnet5large.tf_in1k', pretrained=True)
    model.eval()

    if use_half_precision:
        model = model.half()
        logger.info("Using half precision (FP16) for reduced memory usage")

    _cached_model = model
    return model


def get_model(name):
    assert name == 'pnasnet_large'
    model = _load_model(use_half_precision=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=331, preprocess_type='inception')
    wrapper = PytorchWrapper(identifier='pnasnet_large', model=model,
                             preprocessing=preprocessing,
                             batch_size=8)  # FP16 allows larger batch size
    wrapper.image_size = 331
    return wrapper


def get_layers(name):
    assert name == 'pnasnet_large'
    layer_names = [f'cell_{i + 1}' for i in range(-1, 11)] + ['global_pool']
    return layer_names


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@misc{liu2018progressive,
              title={Progressive Neural Architecture Search},
              author={Chenxi Liu and Barret Zoph and Maxim Neumann and Jonathon Shlens and Wei Hua and Li-Jia Li and Li Fei-Fei and Alan Yuille and Jonathan Huang and Kevin Murphy},
              year={2018},
              eprint={1712.00559},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
            }
            """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
