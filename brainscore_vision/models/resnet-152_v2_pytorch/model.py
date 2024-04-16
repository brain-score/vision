import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torchvision
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of resnet-152_v2.

Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html

Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 

'''

MODEL = torchvision.models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V2')  # use V2 weights


def get_model():
    model_identifier = "resnet-152_v2_pytorch"
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=model_identifier, model=MODEL, preprocessing=preprocessing)
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
    return """@misc{he2016identity,
              title={Identity Mappings in Deep Residual Networks}, 
              author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
              year={2016},
              eprint={1603.05027},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
            """