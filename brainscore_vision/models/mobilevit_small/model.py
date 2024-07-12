from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_images, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import ssl
from transformers import MobileViTForImageClassification
import functools

ssl._create_default_https_context = ssl._create_unverified_context

'''
Can be found on huggingface: https://huggingface.co/apple/mobilevit-small
'''

def get_model(name):
    assert name == 'mobilevit_small'
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    preprocessing = functools.partial(load_preprocess_images, image_size=256)
    wrapper = PytorchWrapper(identifier='mobilevit_small', model=model,
                             preprocessing=preprocessing,
                             batch_size=4)
    wrapper.image_size = 256
    return wrapper


def get_layers(name):
    assert name == 'mobilevit_small'
    layer_names = ["mobilevit.encoder.layer.0", "mobilevit.encoder.layer.1", "mobilevit.encoder.layer.2",
                   "mobilevit.encoder.layer.2.fusion.activation",
                   "mobilevit.encoder.layer.3", "mobilevit.encoder.layer.3.fusion.activation",
                   "mobilevit.encoder.layer.4", "mobilevit.encoder.layer.4.fusion.activation"]

    return layer_names


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@inproceedings{vision-transformer,
                title = {MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
                author = {Sachin Mehta and Mohammad Rastegari},
                year = {2022},
                URL = {https://arxiv.org/abs/2110.02178}
                }
             """


if __name__ == '__main__':
    check_models.check_base_models(__name__)