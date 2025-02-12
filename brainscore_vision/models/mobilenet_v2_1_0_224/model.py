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
    https://huggingface.co/Matthijs/mobilenet_v2_1.0_224
    
Disclaimer: This (pytorch) implementation's Brain-Score scores might not align identically with Tensorflow 
implementation. 
'''


def get_model(name):
    assert name == 'mobilenet_v2_1_0_224'
    model = MobileNetV2ForImageClassification.from_pretrained("Matthijs/mobilenet_v2_1.0_224")

    # this mobilenet was trained with 1001 classes where index 0 is the background class
    # (https://huggingface.co/docs/transformers/en/model_doc/mobilenet_v2)
    classifier_layer = model.classifier
    classifier_layer.register_forward_hook(lambda _layer, _input, logits: logits[:, 1:])

    preprocessing = functools.partial(load_preprocess_images, image_size=224, preprocess_type='inception')
    wrapper = PytorchWrapper(identifier='mobilenet_v2_1_0_224', model=model,
                             preprocessing=preprocessing,
                             batch_size=4)  # doesn't fit into 12 GB GPU memory otherwise
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'mobilenet_v2_1_0_224'
    layer_names = (['mobilenet_v2.conv_stem'] +
                   [f'mobilenet_v2.layer.{i}' for i in range(16)] +
                   ['mobilenet_v2.pooler', 'classifier'])
    return layer_names


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
