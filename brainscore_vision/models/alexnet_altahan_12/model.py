"""alexnet_altahan_12: AlexNet with anti-aliased pooling, fine-tuned on ImageNet with adversarial examples and a
contrastive auxiliary loss.

torchvision AlexNet (61,100,840 parameters) with its three max-pooling layers replaced by anti-aliased max-blur pooling
(Zhang 2019: a stride-1 3x3 max, then a fixed 3x3 binomial blur with stride 2 and reflect padding; no parameters),
fine-tuned from the ImageNet-1k weights on ImageNet-1k for 5.12M images: SGD with momentum 0.9, batch 256, peak lr 5e-3
on a cosine schedule with layer-wise lr decay 0.6 from the top (conv1 at 0.13x), weight decay 1e-3 on conv1-fc7 and
5e-4 on fc8. The cross-entropy batch is a random-resized crop (area 0.25-1) with horizontal flip and 20% grayscale,
replaced by 2-step L2 PGD adversarial versions (eps 0.25 in [0,1] pixel space). An auxiliary SimCLR InfoNCE loss
(weight 1, temperature 0.1) on a 2-layer projection of pool5 is computed on two further augmented views of each
image; the projection head is not part of this model, and the state dict is torchvision's. Read out with
Brain-Score's committed 'alexnet' layer map (V1 features.2, V2 and V4 features.7, IT features.12); behaviour from fc7
(classifier.5); the logits are classifier.6, the output of forward().
"""
import functools

import torch
import torch.nn.functional as F
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_12'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_12.pt'
WEIGHTS_VERSION_ID = 'NBTHTFuB6Dr51D4HTFrWVR6O8gBxv_pZ'
WEIGHTS_SHA1 = '435a696eff1ba68bc0ee0dca15f7d2f6891a99bc'   # checked after download (load_file does not verify it)

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


# Anti-aliased pooling (Zhang 2019): AlexNet's MaxPool2d(3, 2) layers become a stride-1 max followed by a fixed
# 3x3 binomial blur with stride 2. Same indices in `features`, same output sizes (27, 13, 6), no parameters.
MAXPOOL_INDICES = (2, 5, 12)


class MaxBlurPool(torch.nn.Module):
    """MaxPool2d(3, stride 1), then a depthwise [1,2,1]x[1,2,1]/16 blur with stride 2 (reflect padding 1)."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.max = torch.nn.MaxPool2d(kernel_size, stride=1)
        b = torch.tensor([1., 2., 1.])
        # Non-persistent: the state dict stays identical to torchvision's AlexNet.
        self.register_buffer('blur', (b[:, None] * b[None, :] / 16).view(1, 1, 3, 3), persistent=False)

    def forward(self, x):
        x = F.pad(self.max(x), (1, 1, 1, 1), mode='reflect')
        return F.conv2d(x, self.blur.to(x.dtype).expand(x.shape[1], 1, 3, 3), stride=2, groups=x.shape[1])


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
    for i in MAXPOOL_INDICES:
        assert isinstance(model.features[i], torch.nn.MaxPool2d)
        model.features[i] = MaxBlurPool(model.features[i].kernel_size)
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
