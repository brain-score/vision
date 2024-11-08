from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import timm
from brainscore_vision.model_helpers.check_submission import check_models

ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of inception_v4.

Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://huggingface.co/docs/timm/en/models/inception-v4
    
Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 

'''


MODEL = timm.create_model('inception_v4', pretrained=True)

def get_model(name):
    assert name == 'inception_v4_pytorch'
    preprocessing = functools.partial(load_preprocess_images, image_size=299)
    wrapper = PytorchWrapper(identifier='inception_v4_pytorch', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = 299
    return wrapper


def get_layers(name):
    assert name == 'inception_v4_pytorch'
    layers = [] 
    layers += ['Conv2d_1a_3x3']
    layers += ['Mixed_3a']
    layers += ['Mixed_4a']
    layers += [f'Mixed_5{i}' for i in ['a', 'b', 'c', 'd', 'e']]
    layers += [f'Mixed_6{i}' for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
    layers += [f'Mixed_7{i}' for i in ['a', 'b', 'c', 'd']]
    layers += ['global_pool']
    
    return layers


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@misc{szegedy2016inceptionv4,
              title={Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning}, 
              author={Christian Szegedy and Sergey Ioffe and Vincent Vanhoucke and Alex Alemi},
              year={2016},
              eprint={1602.07261},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
            """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
