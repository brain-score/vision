"""alexnet_altahan_09: AlexNet fine-tuned on ImageNet with adversarial examples and dead-leaves occlusions.

torchvision AlexNet (61,100,840 parameters), fine-tuned from the ImageNet-1k weights on ImageNet-1k for 5.12M images
(SGD, lr 1e-3 with a cosine schedule, momentum 0.9, batch 256, random-resized-crop and horizontal flip). Three changes
to plain fine-tuning: the conv weights are pulled toward their pretrained values by an L2-SP penalty (alpha 0.01) in
place of weight decay; half of each mini-batch is replaced by 1-step L2 PGD adversarial versions of the same images
(eps 0.25 in normalised pixel space); and random erasing (p 0.5, area 0.02-0.25, aspect 0.3-3.3) fills each erased
box with a procedurally generated dead-leaves texture of random-colour disks. The state dict is torchvision's. Read
out with Brain-Score's committed 'alexnet' layer map (V1 features.2, V2 features.7, IT features.12) except V4, which
reads the conv3 output (features.7) average-pooled to 6x6 by a parameter-free tap; behaviour from fc7
(classifier.5); the logits are classifier.6, the output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_09'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_09.pt'
WEIGHTS_VERSION_ID = '8vnMcIiT1JuJHSD.6.OY527nudE1xOBd'
WEIGHTS_SHA1 = '024a866b85d6302e84f9fc3042db2607e94164e7'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

IMAGE_SIZE = 224
LAYERS = ['features.2', 'features.5', 'features.7', 'v4_pool', 'features.9', 'features.12',
          'classifier.2', 'classifier.5']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json).
# Set to None to let Brain-Score's LayerSelection choose layers on the public dev benchmarks.
# V4 reads features.7 average-pooled to 6x6 (parameter-free 'v4_pool' tap).
REGION_LAYER_MAP = {'V1': 'features.2', 'V2': 'features.7', 'V4': 'v4_pool', 'IT': 'features.12'}
BEHAVIORAL_READOUT_LAYER = 'classifier.5'
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)


class PooledTapAlexNet(torch.nn.Module):
    """AlexNet with a parameter-free tap 'v4_pool': features.7's output adaptively average-pooled to 6x6.
    The tap is registered first so the base modules keep their names (state dict unchanged) and the
    classifier stays the last-registered module. The tap's output is not used downstream."""

    def __init__(self, base):
        super().__init__()
        self.v4_pool = torch.nn.AdaptiveAvgPool2d(6)
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 7:
                self.v4_pool(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.classifier(x)


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
    model = PooledTapAlexNet(model).eval()
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
