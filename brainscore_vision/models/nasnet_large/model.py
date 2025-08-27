import ssl
import functools
import timm
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images


ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of nasnet_large.
Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://huggingface.co/docs/timm/en/models/nasnet

Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 
'''

MODEL = timm.create_model('nasnetalarge', pretrained=True)


def get_model_list():
    return ['nasnet_large']


def get_model(name):
    assert name == 'nasnet_large'
    preprocessing = functools.partial(load_preprocess_images, image_size=331, preprocess_type='inception')
    wrapper = PytorchWrapper(identifier='nasnet_large', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = 331
    return wrapper


def get_layers(name):
    assert name == 'nasnet_large'
    layer_names = ([f'cell_{i + 1}' for i in range(-1, 5)] + ['reduction_cell_0'] +
                   [f'cell_{i + 1}' for i in range(5, 11)] + ['reduction_cell_1'] +
                   [f'cell_{i + 1}' for i in range(11, 17)] + ['global_pool'])
    return layer_names


def get_bibtex(model_identifier):
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


if __name__ == '__main__':
    check_models.check_base_models(__name__)