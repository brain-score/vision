from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import timm

ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of inception_v3.

Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://huggingface.co/docs/timm/en/models/inception-v3
    
Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 

'''


MODEL = timm.create_model('inception_v3', pretrained=True)


def get_model_list():
    return ['inception_v3_pytorch']


def get_model(name):
    assert name == 'inception_v3_pytorch'
    preprocessing = functools.partial(load_preprocess_images, image_size=299)
    wrapper = PytorchWrapper(identifier='inception_v3_pytorch', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = 299
    return wrapper


def get_layers(name):
    assert name == 'inception_v3_pytorch'
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    return layer_names[2:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@article{DBLP:journals/corr/SzegedyVISW15,
              author    = {Christian Szegedy and
                           Vincent Vanhoucke and
                           Sergey Ioffe and
                           Jonathon Shlens and
                           Zbigniew Wojna},
              title     = {Rethinking the Inception Architecture for Computer Vision},
              journal   = {CoRR},
              volume    = {abs/1512.00567},
              year      = {2015},
              url       = {http://arxiv.org/abs/1512.00567},
              archivePrefix = {arXiv},
              eprint    = {1512.00567},
              timestamp = {Mon, 13 Aug 2018 16:49:07 +0200},
              biburl    = {https://dblp.org/rec/journals/corr/SzegedyVISW15.bib},
              bibsource = {dblp computer science bibliography, https://dblp.org}
            }
            """