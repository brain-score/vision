"""alexnet_altahan_06: AlexNet with local response normalisation, fine-tuned on ImageNet with a SimCLR auxiliary,
framed, edge-map and wide-crop training views; V1, V2 and V4 read from anti-aliased passes, IT averaged over small
translations, behaviour read from centred, L2-normalised logits, and label logits averaged over an 8 degree and a 7
degree view.

torchvision AlexNet with the original AlexNet's parameter-free local response normalisation restored after the conv1
and conv2 ReLUs, fine-tuned on ImageNet-1k (SGD, lr 1e-3, momentum 0.9, weight decay 5e-4, batch 256, 5.12M images;
cross-entropy plus a SimCLR NT-Xent auxiliary loss on fc7 through a projection head that is discarded after training).
Training views: random-resized-crop with area scale 0.35-1 and horizontal flip; for the cross-entropy input, 10% of
the images are replaced by their Sobel edge map, and 80% are then shrunk by a factor U[0.35, 1] and pasted centred on
a uniform grey canvas of random level; the SimCLR views are unframed. The LRN layers have no weights, so the network
has AlexNet's 61,100,840 parameters and the state dict is torchvision's. Training used the plain network; the
readouts below are fixed, parameter-free and used only at test time.

Readouts (the hooked module is called last on its readout pass, so its hook holds that pass; nothing is learned):
- V1 `v1_avgpool`: conv1's trained filters applied at stride 2 instead of 4, ReLU, a fixed 3x3 binomial blur with
  stride 2 (`BinomialBlur`, an anti-aliased downsample), conv1's LRN and a 3x3 stride-2 average pool (64x27x27).
- V2 features.7 (conv3 ReLU, 384x13x13): a pass whose front end applies conv1's filters at stride 1, ReLU, the
  binomial blur twice and LRN, then the network's pool1, conv2+ReLU+LRN, pool2 and conv3.
- V4 features.9 (conv4 ReLU, 256x13x13): a pass from the V1 front end through pool1, conv2, pool2, conv3 and conv4.
- IT features.12 (pool5): the mean of pool5 over four copies of the image shifted by 0 or 2 px in x and y (reflect
  padding), which averages out conv1's stride-4 sampling phase.
- Behaviour `logits_centred_l2`: the 1000 logits of the unmodified image, centred and L2-normalised per image.
- forward()'s label logits: the mean of the logits on the 8 degree image and on its central 7/8 crop resized back
  (`CentralZoom`). The extra modules are registered before the base network's, so Brain-Score's default `logits`
  layer is still classifier.6, the output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_06'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_06.pt'
WEIGHTS_VERSION_ID = 'cI5ZJb92.NBzChEw.2YJY1vY_ot32Y7Q'
WEIGHTS_SHA1 = '7aa00dc51e028062b6690b868a1681643a114356'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

IMAGE_SIZE = 224
LAYERS = ['v1_avgpool', 'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5', 'classifier.6', 'logits_centred_l2']
# Brain-Score's committed map for its 'alexnet' plugin (region_layer_map/alexnet.json).
# Set to None to let Brain-Score's LayerSelection choose layers on the public dev benchmarks.
REGION_LAYER_MAP = {'V1': 'v1_avgpool', 'V2': 'features.7', 'V4': 'features.9', 'IT': 'features.12'}
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


class BinomialBlur(torch.nn.Module):
    """Fixed 3x3 binomial ([1,2,1] outer [1,2,1] / 16) depthwise blur with stride 2 and zero padding 1: an
    anti-aliased 2x downsample. No parameters; the kernel is a non-persistent buffer, so the state dict is unchanged."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', (torch.outer(k, k) / 16)[None, None], persistent=False)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.kernel.expand(x.shape[1], 1, 3, 3), stride=2, padding=1,
                                          groups=x.shape[1])


class AlexNetWithLogitTap(torch.nn.Module):
    """AlexNet whose label logits are the mean of the logits on the unmodified image and on a centrally magnified
    copy, while the neural taps and the `logits_centred_l2` tap (output discarded; Brain-Score reads it via a hook)
    see only the unmodified image (the IT tap averages it over four 2 px translations). The zoom and the tap are registered first, so classifier.6 stays the
    last-registered module, and module names and state-dict keys match AlexNet's."""

    def __init__(self, base):
        super().__init__()
        self.zoom = CentralZoom()
        # V1 tap only: conv1's trained filters sampled at stride 2 (not 4), then ReLU, an anti-aliasing binomial
        # blur with stride 2 (64x55x55), conv1's LRN and v1_avgpool; the tap stays 64x27x27.
        self.v1_blur = BinomialBlur()
        # V1 tap: fixed 3x3 stride-2 average pool of the anti-aliased conv1 ReLU+LRN output (see v1_blur), in place
        # of the max pool (features.2); same 64x27x27 shape. Its output is discarded; Brain-Score reads it via a hook.
        self.v1_avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        # V2 tap (features.7) is likewise overwritten in forward() by a separate pass whose front end samples conv1's
        # trained filters at stride 1 (not 4; 64x218x218), ReLU, this fixed binomial blur with stride 2 twice (64x109x109,
        # then 64x55x55) and conv1's LRN. The V4 tap (features.9) is overwritten by a pass from the V1 front end.
        self.v2_blur = BinomialBlur()
        self.logits_centred_l2 = CentredL2()
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def forward(self, x):
        zoomed = self.classifier[:6](torch.flatten(self.avgpool(self.features(self.zoom(x))), 1))
        # Unmodified pass last: a forward hook keeps a module's latest output, so features.* and classifier.2/5
        # hold the 8 deg image's activations.
        conv1 = self.features[:2](x)
        dense = torch.nn.functional.conv2d(x, self.features[0].weight, self.features[0].bias, stride=2, padding=2)
        front = self.features[1][1](self.v1_blur(torch.relu(dense)))   # anti-aliased conv1 ReLU+LRN, 64x55x55
        self.v1_avgpool(front)   # read by its hook only
        hidden = self.classifier[:6](torch.flatten(self.avgpool(self.features[2:](conv1)), 1))
        # V2 tap only: the stride-1, two-blur front end through pool1, conv2+ReLU+LRN, pool2 and conv3, then features.7
        # (conv3 ReLU, 384x13x13) called last, so its hook holds this pass. The pools are applied functionally so the
        # features.2 and features.5 taps keep the unmodified pass; output discarded.
        full = torch.nn.functional.conv2d(x, self.features[0].weight, self.features[0].bias, stride=1, padding=2)
        v2 = self.features[1][1](self.v2_blur(self.v2_blur(torch.relu(full))))   # anti-aliased conv1 ReLU+LRN, 64x55x55
        v2 = torch.nn.functional.max_pool2d(v2, kernel_size=3, stride=2)
        v2 = torch.nn.functional.max_pool2d(self.features[4](self.features[3](v2)), kernel_size=3, stride=2)
        self.features[7](self.features[6](v2))   # read by its hook only
        # V4 tap only: the anti-aliased front end through pool1, conv2+ReLU+LRN, pool2, conv3 and its ReLU, then
        # conv4 and features.9 (conv4 ReLU, 256x13x13) called last, so its hook holds this pass. The pools and conv3's
        # ReLU are applied functionally so the features.2, features.5 and features.7 taps keep their passes;
        # output discarded.
        v4 = torch.nn.functional.max_pool2d(front, kernel_size=3, stride=2)
        v4 = torch.nn.functional.max_pool2d(self.features[4](self.features[3](v4)), kernel_size=3, stride=2)
        v4 = torch.relu(self.features[6](v4))
        self.features[9](self.features[8](v4))   # read by its hook only
        # IT tap only: pool5 averaged over four copies of the unmodified image translated by (0,0), (0,2), (2,0) and
        # (2,2) px (reflect-pad 2 px on the leading edges, crop back), averaging out conv1's stride-4 phase. The pools
        # and the conv3/conv4 ReLUs are applied functionally, so the features.2/5/7/9 hooks keep their passes.
        h, w = x.shape[-2:]
        f = self.features
        s = torch.cat([torch.nn.functional.pad(x, (dx, 0, dy, 0), mode='reflect')[..., :h, :w]
                       for dy, dx in ((0, 0), (0, 2), (2, 0), (2, 2))])
        s = torch.nn.functional.max_pool2d(f[4](f[3](torch.nn.functional.max_pool2d(f[1](f[0](s)), 3, 2))), 3, 2)
        s = f[11](f[10](torch.relu(f[8](torch.relu(f[6](s))))))
        it = torch.nn.functional.max_pool2d(s, 3, 2).unflatten(0, (4, -1)).mean(0)
        # features.12 (3x3 stride-2 max pool) called last on a 13x13 map that holds the mean at each window's centre
        # (odd indices, each in exactly one window) and -inf elsewhere, so it outputs exactly the mean and its hook
        # (which returns None) holds it; output discarded.
        grid = torch.full_like(s[:x.shape[0]], float('-inf'))
        grid[..., 1::2, 1::2] = it
        f[12](grid)   # read by its hook only
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
