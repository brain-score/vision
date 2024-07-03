from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import timm
from brainscore_vision.model_helpers.check_submission import check_models

ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of densenet_201.

Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://huggingface.co/timm/densenet201.tv_in1k
    
Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 

'''


MODEL = timm.create_model('densenet201.tv_in1k', pretrained=True)


def get_model(name):
    assert name == 'densenet_201_pytorch'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='densenet_201_pytorch', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'densenet_201_pytorch'
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    return layer_names[2:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@inproceedings{huang2017densely,
              title={Densely Connected Convolutional Networks},
              author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
              year={2017}
            }
            """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
