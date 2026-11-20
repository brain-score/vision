"""alexnet_altahan_10: AlexNet adversarially fine-tuned on ImageNet with high-pass PGD steps, read at 256 px.

torchvision AlexNet (61,100,840 parameters), fine-tuned from the ImageNet-1k weights on ImageNet-1k for 5.12M images
(SGD, lr 1e-3 with a cosine schedule, momentum 0.9, weight decay 5e-4, batch 256, random-resized-crop and horizontal
flip) on adversarial examples only: L2 PGD with eps 0.5 in [0, 1] pixel units, 3 steps of 0.4167, where every step
direction has its own Gaussian blur (sigma 2 px, 13x13, reflect padding) subtracted and is renormalised to the step
size before the projection, so the whole perturbation budget goes to high spatial frequencies. The state dict is
torchvision's. Images are presented at 256 px over 7.75 degrees. Read out with Brain-Score's committed 'alexnet' layer
map (V1 features.2, V4 features.7, IT features.12) except V2, which reads a parameter-free 7x7 / stride-2 max pool of
the conv2 ReLU output (v2_maxpool7, the same 15x15 grid as pool2); behaviour from fc7 (classifier.5); the logits are
classifier.6, the output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_10'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_10.pt'
WEIGHTS_VERSION_ID = 'qld.hpCsLwmUOeJjjOrNqqbQjpgITekR'
WEIGHTS_SHA1 = '60ea9794f006d93309ee12a4761550f9bd8c231a'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

IMAGE_SIZE = 256            # 256 px over 7.75 deg: about 33 px/deg
LAYERS = ['features.2', 'features.5', 'v2_maxpool7', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json), except V2.
# V2 reads MaxPool(k=7, s=2, p=2) of conv2's ReLU output (features.4): the same 15x15x192 grid as pool2 (k=3).
REGION_LAYER_MAP = {'V1': 'features.2', 'V2': 'v2_maxpool7', 'V4': 'features.7', 'IT': 'features.12'}
BEHAVIORAL_READOUT_LAYER = 'classifier.5'
VISUAL_DEGREES = 7.75         # field of view


class WideMaxPoolAlexNet(torch.nn.Module):
    """AlexNet with a parameter-free tap `v2_maxpool7`: MaxPool(k=7, s=2, p=2) applied to features.4 (ReLU(conv2)),
    a wider max than pool2's k=3 on the same 15x15 grid. The tap is a side branch; the main path still runs
    features.5 (k=3), so the logits are unchanged. Submodule names match torchvision's AlexNet, so the state dict
    is unchanged; the tap is registered first, so the classifier stays the last-registered module."""
    def __init__(self, base):
        super().__init__()
        self.v2_maxpool7 = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=2)
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 4:
                self.v2_maxpool7(x)
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
    # conv2's ReLU is in-place, which would overwrite conv2's output (features.3); make it out-of-place.
    model.features[4].inplace = False
    model = WideMaxPoolAlexNet(model).eval()
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
