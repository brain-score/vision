from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import timm

ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of nasnet_large.

Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://huggingface.co/docs/timm/en/models/nasnet
    
Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 

'''


MODEL = timm.create_model('nasnetalarge', pretrained=True)


def get_model():
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='nasnet_large_pytorch', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = 224
    return wrapper


def get_layers():
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    return layer_names[2:]


def get_bibtex():
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@misc{zoph2018learning,
              title={Learning Transferable Architectures for Scalable Image Recognition},
              author={Barret Zoph and Vijay Vasudevan and Jonathon Shlens and Quoc V. Le},
              year={2018},
              eprint={1707.07012},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
            """