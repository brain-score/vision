"""alexnet_altahan_05: AlexNet with batch normalisation and a cosine-softmax classifier, trained from random
initialisation on ImageNet with 1/n eigenspectrum priors on two layers; fc7 and the logits are averaged over the image
and its mirror.

torchvision AlexNet with BatchNorm2d after every convolution (each conv features.N becomes Sequential(conv, BN), so
every other module name is torchvision's) and a bias-free cosine-softmax classifier, logits = (W/||W||_row) . (h/||h||)
/ 0.05, which gives 61,102,144 parameters. Trained from random initialisation on ImageNet-1k only: 5.12M images (40k
steps at batch 128), SGD with momentum 0.9 and weight decay 5e-4, peak lr 0.01 reached by a linear warmup over the
first 5% of the images and then cosine-decayed to 0, random-resized-crop with area scale 0.35-1 and horizontal flip.
Loss: cross-entropy plus, with weight 0.1 each, a 1/n power-law prior (Nassar et al. 2020) on the top 64 eigenvalues
of the batch covariance of the globally average-pooled conv3 ReLU (features.7) and pool5 (features.12) outputs.
fc7 (classifier.5) and the logits (classifier.6) are the mean over the image and its horizontal mirror, in training
and at test time; the features.* and classifier.0-4 hooks hold the unmirrored image's activations. Read out with
Brain-Score's committed 'alexnet' layer map (V1 features.2, V2 and V4 features.7, IT features.12); behaviour from the
flip-averaged logits (classifier.6). The wrapper adds no modules, so Brain-Score's default `logits` layer (the
last-registered module) is classifier.6, the output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_05'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_05.pt'
WEIGHTS_VERSION_ID = 'ccxNj0XGN3Kcm4j0NZJG9oiwbPZFCXx7'
WEIGHTS_SHA1 = 'cf6c1fa23033f3d51ebc62dc9586fe1d2fa493d9'   # checked after download (load_file does not verify it)

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
          'classifier.2', 'classifier.5', 'classifier.6']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json).
# Set to None to let Brain-Score's LayerSelection choose layers on the public dev benchmarks.
REGION_LAYER_MAP = {'V1': 'features.2', 'V2': 'features.7', 'V4': 'features.7', 'IT': 'features.12'}
BEHAVIORAL_READOUT_LAYER = 'classifier.6'    # the cosine-softmax logits, averaged over the image and its mirror (see FlipAveragedAlexNet)
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)

TEMPERATURE = 0.05


class CosineHead(torch.nn.Module):
    """Bias-free cosine-softmax classifier (Kornblith 2021): logits = (W/||W||_row) . (h/||h||) / TEMPERATURE."""
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())

    def forward(self, h, h2=None):
        """Logits of h, or with h2 (a second view) the mean of the two views' logits, in one call so a forward hook
        sees the mean."""
        with torch.autocast(h.device.type, enabled=False):   # fp32: logits span +-20, fp16 cosines are too coarse
            w = torch.nn.functional.normalize(self.weight.float(), dim=1)
            u = torch.nn.functional.normalize(h.float(), dim=1)
            if h2 is not None:
                u = (u + torch.nn.functional.normalize(h2.float(), dim=1)) / 2
            return u @ w.T / TEMPERATURE


class FlipAveragedAlexNet(torch.nn.Module):
    """AlexNet whose fc7 (classifier.5) and logits (classifier.6) are the mean over the image and its horizontal
    mirror, while features.* and classifier.0-4 see only the original image. Same module names and state dict as
    the base network, and no new modules, so classifier.6 stays the last-registered module."""

    def __init__(self, net):
        super().__init__()
        self.features, self.avgpool, self.classifier = net.features, net.avgpool, net.classifier

    def fc7(self, x):
        return self.classifier[:6](torch.flatten(self.avgpool(self.features(x)), 1))

    def forward(self, x):
        h_flip = self.fc7(torch.flip(x, dims=[3]))
        # Original image second: a forward hook keeps a module's latest output, so features.* (V1, V2, V4, IT) and
        # classifier.0-4 hold the original image's activations.
        h = self.fc7(x)
        # classifier.5 is a ReLU; the mean of two non-negative fc7s passes through it unchanged, so calling it once
        # more leaves its hook holding the two views' mean fc7 (the behavioral readout).
        self.classifier[5]((h + h_flip) / 2)
        return self.classifier[6](h, h_flip)


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    model = torchvision.models.alexnet(weights=None)
    # BatchNorm after every conv: features.N becomes Sequential(conv, BN), so the conv's random init is the root's
    # and every other features.N index (and REGION_LAYER_MAP) is unchanged.
    for i, m in enumerate(model.features):
        if isinstance(m, torch.nn.Conv2d):
            model.features[i] = torch.nn.Sequential(m, torch.nn.BatchNorm2d(m.out_channels))
    # classifier.6 becomes a cosine-softmax head initialised from the root's classifier.6 weights (its bias dropped).
    model.classifier[6] = CosineHead(model.classifier[6].weight)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = FlipAveragedAlexNet(model).eval()
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
