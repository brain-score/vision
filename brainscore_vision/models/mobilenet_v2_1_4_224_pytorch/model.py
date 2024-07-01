from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_images, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import ssl
from transformers import MobileNetV2ForImageClassification
import functools

ssl._create_default_https_context = ssl._create_unverified_context

'''
This is a Pytorch implementation of mobilenet_v2_1.0_224.

Previously on Brain-Score, this model existed as a Tensorflow model, and was converted via:
    https://huggingface.co/Matthijs/mobilenet_v2_1.4_224
    
Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 

'''


MODEL = MobileNetV2ForImageClassification.from_pretrained("Matthijs/mobilenet_v2_1.4_224")


def get_model(name):
    assert name == 'mobilenet_v2_1-4_224_pytorch'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='mobilenet_v2_1-4_224_pytorch', model=MODEL,
                             preprocessing=preprocessing,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'mobilenet_v2_1-4_224_pytorch'
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    return layer_names[-50:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@inproceedings{mobilenetv22018,
                title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
                author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
                booktitle={CVPR},
                year={2018}
                }
            """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
