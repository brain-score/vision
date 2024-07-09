from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_images, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import ssl
from torchvision import models
import functools

ssl._create_default_https_context = ssl._create_unverified_context

'''
Can be found in PyTorch models: https://pytorch.org/vision/main/models/generated/torchvision.models.shufflenet_v2_x2_0.html
'''

MODEL = models.shufflenet_v2_x2_0(weights='IMAGENET1K_V1')


def get_model(name):
    assert name == 'shufflenet_v2'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='shufflenet_v2', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'shufflenet_v2'
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    return layer_names[2:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@misc{ma2018shufflenetv2practicalguidelines,
                  title={ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design}, 
                  author={Ningning Ma and Xiangyu Zhang and Hai-Tao Zheng and Jian Sun},
                  year={2018},
                  eprint={1807.11164},
                  archivePrefix={arXiv},
                  primaryClass={cs.CV},
                  url={https://arxiv.org/abs/1807.11164}, 
                }
              """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
