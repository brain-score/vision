"""vgg_16_altahan_02: torchvision's ImageNet-pretrained VGG-16 (no batch norm), unmodified, read out with Brain-Score's
committed 'vgg_16' layer map.

The weights are torchvision's IMAGENET1K_V1 (downloaded by torchvision; no upload). V1, V2 and V4 read pool3
(features.16), IT reads pool4 (features.23), and behaviour reads fc7 (classifier.3, the last listed layer); 8 degrees,
224 px. Brain-Score's default `logits` layer is classifier.6, the output of forward().
"""
import functools

import torchvision

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'vgg_16_altahan_02'

BIBTEX = """@misc{simonyan2015deepconvolutionalnetworkslargescale,
                  title = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
                  author = {Karen Simonyan and Andrew Zisserman},
                  year = {2015},
                  eprint = {1409.1556},
                  archivePrefix = {arXiv},
                  primaryClass = {cs.CV},
                  url = {https://arxiv.org/abs/1409.1556}
                  }"""

IMAGE_SIZE = 224
# Brain-Score's 'vgg_16' plugin: its layer list and committed region_layer_map/vgg_16.json. Behavior reads the last
# listed layer (fc7, classifier.3), as Brain-Score does when no readout layer is given.
LAYERS = ['features.4', 'features.9', 'features.16', 'features.23', 'features.30', 'classifier.0', 'classifier.3']
REGION_LAYER_MAP = {'V1': 'features.16', 'V2': 'features.16', 'V4': 'features.16', 'IT': 'features.23'}
BEHAVIORAL_READOUT_LAYER = 'classifier.3'
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    model = torchvision.models.vgg16(weights='IMAGENET1K_V1').eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return LAYERS


def get_bibtex(model_identifier):
    return BIBTEX


if __name__ == '__main__':
    check_models.check_base_models(__name__)
