from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import timm

ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a half-precision (FP16) Pytorch implementation of pnasnet_large.

Based on the original pnasnet_large implementation, with FP16 for reduced memory usage.
See: https://huggingface.co/timm/pnasnet5large.tf_in1k

'''

MODEL = timm.create_model('pnasnet5large.tf_in1k', pretrained=True).half()


def get_model(name):
    assert name == 'pnasnet_large_half'
    preprocessing = functools.partial(load_preprocess_images, image_size=331, preprocess_type='inception')
    wrapper = PytorchWrapper(identifier='pnasnet_large_half', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=8)  # FP16 allows larger batch size
    wrapper.image_size = 331
    return wrapper


def get_layers(name):
    assert name == 'pnasnet_large_half'
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
