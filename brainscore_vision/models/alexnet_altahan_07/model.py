"""alexnet_altahan_07: AlexNet with random initialisation and no training.

torchvision AlexNet (61,100,840 parameters) with PyTorch's default initialisation, drawn once from a fixed seed; no
weight has been trained. The weights file holds that initialisation, so the network does not depend on the random
number generator of the installed library versions. Read out with Brain-Score's committed 'alexnet' layer map
(V1 features.2, V2 and V4 features.7, IT features.12) and behaviour from fc7 (classifier.5); the logits are
classifier.6, the output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_07'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_07.pt'
WEIGHTS_VERSION_ID = 'j_z0CmLq6qNdWDgxQ14TE_S0ju0ymLly'
WEIGHTS_SHA1 = '9d5443511a36bfb94e10979ab11342eff898ac36'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

IMAGE_SIZE = 224
LAYERS = ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json).
# Set to None to let Brain-Score's LayerSelection choose layers on the public dev benchmarks.
REGION_LAYER_MAP = {'V1': 'features.2', 'V2': 'features.7', 'V4': 'features.7', 'IT': 'features.12'}
BEHAVIORAL_READOUT_LAYER = 'classifier.5'
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    model = torchvision.models.alexnet(weights=None)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = model.eval()
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
