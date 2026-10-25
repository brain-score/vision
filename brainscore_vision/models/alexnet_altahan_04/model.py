"""alexnet_altahan_04: AlexNet with local response normalisation, fine-tuned on ImageNet with a SimCLR auxiliary,
framed, edge-map and wide-crop training views; V1 and V4 read through fixed average pools, behaviour read from
centred, L2-normalised logits, and label logits averaged over an 8 degree and a 7 degree view.

torchvision AlexNet with the original AlexNet's parameter-free local response normalisation restored after the conv1
and conv2 ReLUs, fine-tuned on ImageNet-1k (SGD, lr 1e-3, momentum 0.9, weight decay 5e-4, batch 256, 5.12M images;
cross-entropy plus a SimCLR NT-Xent auxiliary loss on fc7 through a projection head that is discarded after training).
Training views: random-resized-crop with area scale 0.35-1 and horizontal flip; for the cross-entropy input, 10% of
the images are replaced by their Sobel edge map, and 80% are then shrunk by a factor U[0.35, 1] and pasted centred on
a uniform grey canvas of random level; the SimCLR views are unframed. The LRN layers have no weights, so the network
has AlexNet's 61,100,840 parameters and the state dict is torchvision's. Read out with Brain-Score's committed
'alexnet' layer map except V1 and V4, which are read through parameter-free average pools: `v1_avgpool` is a 3x3
stride-2 average pool of the conv1 ReLU+LRN output (in place of the first max pool, 64x27x27), `v4_avgpool` a 3x3
stride-1 average pool (padding 1, padding excluded from the mean) of the conv4 ReLU output (256x13x13). Behaviour is
read from the 1000 logits, centred and L2-normalised per image by the parameter-free `logits_centred_l2` module.
forward()'s label logits are the mean of the logits on the 8 degree image and on its central 7/8 crop resized back
(`CentralZoom`, nothing learned); the neural and behavioural taps see only the unmodified image. The taps and the zoom
are registered before the base network's modules, so Brain-Score's default `logits` layer is still classifier.6, the
output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_04'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_04.pt'
WEIGHTS_VERSION_ID = '6_aPJFWA393YzSzRWjwHqN5ux8qvKS40'
WEIGHTS_SHA1 = '70c12767024ddc4f83e49b60ffe43406c52f6fc3'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

IMAGE_SIZE = 224
LAYERS = ['v1_avgpool', 'v4_avgpool', 'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5', 'classifier.6', 'logits_centred_l2']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json).
# Set to None to let Brain-Score's LayerSelection choose layers on the public dev benchmarks.
REGION_LAYER_MAP = {'V1': 'v1_avgpool', 'V2': 'features.7', 'V4': 'v4_avgpool', 'IT': 'features.12'}
BEHAVIORAL_READOUT_LAYER = 'logits_centred_l2'   # the 1000 logits, centred and L2-normalised per image (parent: 'classifier.6')
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)

LABEL_DEGREES = 7   # forward()'s logits average the VISUAL_DEGREES view with this magnified view


class CentredL2(torch.nn.Module):
    """Per-image (l - mean(l)) / ||l - mean(l)||_2 over the last dim. No parameters."""

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class CentralZoom(torch.nn.Module):
    """Fixed, parameter-free magnification: keep the central LABEL_DEGREES / VISUAL_DEGREES of the image
    (196 of 224 px) and resize it back with antialiased bilinear interpolation. Resizing is a per-channel convex
    combination, so it commutes with the normalization and applying it to the normalized input equals zooming
    the raw image."""

    def forward(self, x):
        h, w = x.shape[-2:]
        ch, cw = round(h * LABEL_DEGREES / VISUAL_DEGREES), round(w * LABEL_DEGREES / VISUAL_DEGREES)
        top, left = (h - ch) // 2, (w - cw) // 2
        x = x[..., top:top + ch, left:left + cw]
        return torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=True)


class AlexNetWithLogitTap(torch.nn.Module):
    """AlexNet whose label logits are the mean of the logits on the unmodified image and on a centrally magnified
    copy, while the neural taps and the `logits_centred_l2` tap (output discarded; Brain-Score reads it via a hook)
    see only the unmodified image. The zoom and the tap are registered first, so classifier.6 stays the
    last-registered module, and module names and state-dict keys match AlexNet's."""

    def __init__(self, base):
        super().__init__()
        self.zoom = CentralZoom()
        # V1 tap: fixed 3x3 stride-2 average pool of the conv1 ReLU+LRN output (features.1), in place of the max
        # pool (features.2); same 64x27x27 shape. Its output is discarded; Brain-Score reads it via a hook.
        self.v1_avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        # V4 tap: fixed 3x3 stride-1 average pool (padding 1, padding excluded from the mean) of the conv4 ReLU
        # (features.9); same 384x13x13 shape. Its output is discarded; Brain-Score reads it via a hook.
        self.v4_avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        self.logits_centred_l2 = CentredL2()
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def forward(self, x):
        zoomed = self.classifier[:6](torch.flatten(self.avgpool(self.features(self.zoom(x))), 1))
        # Unmodified pass last: a forward hook keeps a module's latest output, so features.* and classifier.2/5
        # hold the 8 deg image's activations.
        conv1 = self.features[:2](x)
        self.v1_avgpool(conv1)   # read by its hook only
        conv4 = self.features[2:10](conv1)
        self.v4_avgpool(conv4)   # read by its hook only
        hidden = self.classifier[:6](torch.flatten(self.avgpool(self.features[10:](conv4)), 1))
        last = self.classifier[6]
        self.logits_centred_l2(torch.nn.functional.linear(hidden, last.weight, last.bias))   # read by its hook only
        # classifier.6 is linear, so applying it to the mean of the two fc7 vectors gives the mean of the two
        # views' logits; called once, its tap and forward() both hold the averaged logits.
        return last((hidden + zoomed) / 2)


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    model = torchvision.models.alexnet(weights=None)
    # Original AlexNet's parameter-free local response normalisation after the conv1 and conv2 ReLUs. Each ReLU
    # becomes Sequential(ReLU, LRN) in place, so features.2/5/... keep their names and the state dict is unchanged.
    for i in (1, 4):
        model.features[i] = torch.nn.Sequential(model.features[i],
                                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = AlexNetWithLogitTap(model).eval()
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
