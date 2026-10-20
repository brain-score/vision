"""alexnet_altahan_01_v2: AlexNet fine-tuned on ImageNet, read out at three visual scales.

Weights: torchvision AlexNet fine-tuned on ImageNet-1k (SGD, lr 3e-3, 5.12M images); nothing else is trained.
Readout: stimuli are placed at 8 degrees and loaded once at 304 px. Every region's view is a centre crop of that
load, resized to 224 px: V1 reads the full load (8 degrees), V2 and behavior its central 228 px (6 degrees, taps
under 'deg6.'), V4 and IT its central 190 px (5 degrees, taps under 'deg5.'). The crop passes are
parameter-free, so the network has AlexNet's 61,100,840 parameters.
"""
import functools

import torch
import torch.nn.functional as F
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_01_v2'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_01.pt'
WEIGHTS_VERSION_ID = '05EKjqQPjdxQxrdwJhHimbxo1q8uUyUR'
WEIGHTS_SHA1 = '21993053035779d0787ba9e99a9ac98775a102d3'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

NET_SIZE = 224                # the size AlexNet is run at, in every pass
IMAGE_SIZE = 304              # stimulus load size: its central 6/8 (228 px) and 5/8 (190 px) are exact
LAYERS = ['features.2', 'deg6.features_7', 'deg6.classifier_2', 'deg5.features_7', 'deg5.features_12']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json), each region read
# from the same layer of its own view: V1 at 8 degrees, V2 at 6, V4 and IT at 5.
REGION_LAYER_MAP = {'V1': 'features.2', 'V2': 'deg6.features_7', 'V4': 'deg5.features_7',
                    'IT': 'deg5.features_12'}
BEHAVIORAL_READOUT_LAYER = 'deg6.classifier_2'  # fc6 (ReLU after the first FC layer), 6-degree view
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)


def _resize(x):
    if x.shape[-1] == NET_SIZE and x.shape[-2] == NET_SIZE:
        return x
    return F.interpolate(x, size=(NET_SIZE, NET_SIZE), mode='bilinear', align_corners=False, antialias=True)


def _crop(x, degrees):
    size = x.shape[-1]
    crop = round(size * degrees / VISUAL_DEGREES)
    top = (size - crop) // 2
    return x[..., top:top + crop, top:top + crop]


class MultiScaleAlexNet(torch.nn.Module):
    """The loaded AlexNet's own modules (same names and state dict), plus parameter-free passes on centre
    crops of the input. The crop passes call each layer's .forward directly, so Brain-Score's hooks on the
    shared layers fire only for the full 8-degree pass; the crop passes' activations are exposed through
    Identity taps instead. Returns the full pass's logits."""
    TAPS = {'deg6': {'features': {7: 'features_7'}, 'classifier': {2: 'classifier_2'}},
            'deg5': {'features': {7: 'features_7', 12: 'features_12'}, 'classifier': {}}}
    DEGREES = {'deg6': 6, 'deg5': 5}

    def __init__(self, net):
        super().__init__()
        # Taps are registered before AlexNet's modules so that the classifier is the last-registered module, where
        # Brain-Score's PytorchWrapper looks for the 'logits' that label benchmarks read.
        for view, taps in self.TAPS.items():
            self.add_module(view, torch.nn.ModuleDict({name: torch.nn.Identity()
                                                       for t in taps.values() for name in t.values()}))
        self.features, self.avgpool, self.classifier = net.features, net.avgpool, net.classifier

    def _unhooked(self, seq, h, taps, view):
        for i, layer in enumerate(seq):
            h = layer.forward(h)
            if i in taps:
                h = self._modules[view][taps[i]](h)
        return h

    def forward(self, x):
        for view, taps in self.TAPS.items():
            h = self._unhooked(self.features, _resize(_crop(x, self.DEGREES[view])), taps['features'], view)
            if taps['classifier']:
                self._unhooked(self.classifier, torch.flatten(self.avgpool.forward(h), 1), taps['classifier'], view)
        return self.classifier(torch.flatten(self.avgpool(self.features(_resize(x))), 1))



def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    net = torchvision.models.alexnet(weights=None)
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = MultiScaleAlexNet(net).eval()
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
