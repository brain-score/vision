"""vgg_16_altahan_01: VGG-16 with local response normalisation before pool1 and pool2, fine-tuned on ImageNet with a
SimCLR auxiliary and framed, edge-map and wide-crop training views; V4 read from conv4_1 and behaviour from centred,
L2-normalised logits.

torchvision VGG-16 (no batch norm) with a parameter-free local response normalisation (size 5, alpha 1e-4, beta 0.75,
k 2) after the ReLU before pool1 and pool2, fine-tuned on ImageNet-1k (SGD, lr 1e-3, momentum 0.9, weight decay 5e-4,
batch 256, 5.12M images; cross-entropy plus a SimCLR NT-Xent auxiliary loss on fc7 through a projection head that is
discarded after training). Training views: random-resized-crop with area scale 0.35-1 and horizontal flip; for the
cross-entropy input, 10% of the images are replaced by their Sobel edge map, and 80% are then shrunk by a factor
U[0.35, 1] and pasted centred on a uniform grey canvas of random level; the SimCLR views are unframed. The LRN layers have no weights, so the network has VGG-16's
138,357,544 parameters and the state dict is torchvision's.

Read out with Brain-Score's committed 'vgg_16' layer map (V1 and V2 pool3 = features.16, IT pool4 = features.23)
except V4, which is read from conv4_1's ReLU (features.18); behaviour from the 1000 logits, centred and L2-normalised
per image by the parameter-free `logits_centred_l2` module. The model code is shared with variants that switch on
further parameter-free readout passes through STEP; at STEP = 8 none of them runs, so forward() is the plain network
plus that tap. The extra modules are registered before the base network's, so Brain-Score's default `logits` layer is
classifier.6, the output of forward().
"""
import functools

import torch
import torch.utils.checkpoint
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'vgg_16_altahan_01'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/vgg_16_altahan_01.pt'
WEIGHTS_VERSION_ID = '9DW3aiO5mDBOjJ_RP3clNpRVqJhnml_Y'
WEIGHTS_SHA1 = 'f97dca311652a75b6d4d680602f2c1f4ef57755c'   # checked after download (load_file does not verify it)

BIBTEX = """@misc{simonyan2015deepconvolutionalnetworkslargescale,
                  title = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
                  author = {Karen Simonyan and Andrew Zisserman},
                  year = {2015},
                  eprint = {1409.1556},
                  archivePrefix = {arXiv},
                  primaryClass = {cs.CV},
                  url = {https://arxiv.org/abs/1409.1556}
                  }"""

STEP = 8
IMAGE_SIZE = 224
REGION_LAYER_MAP = {'V1': 'v1_avgpool' if STEP >= 10 else 'features.16', 'V2': 'features.16',
                    'V4': 'features.18' if STEP >= 2 else 'features.16', 'IT': 'features.23'}
BEHAVIORAL_READOUT_LAYER = 'logits_centred_l2' if STEP >= 5 else 'classifier.6' if STEP >= 4 else 'classifier.3'
LAYERS = list(dict.fromkeys(['features.4', 'features.9', 'features.16', 'features.23', 'features.30', 'classifier.0',
                             'classifier.3', *REGION_LAYER_MAP.values(), BEHAVIORAL_READOUT_LAYER]))
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)
LABEL_DEGREES = 7             # step 9: forward()'s logits average the VISUAL_DEGREES view with this magnified view

TAPPED = {16: 'pool', 18: 'relu', 23: 'pool'}   # features.* tap modules; extra passes apply these functionally
# Rows per features() call. Training batches carry the SimCLR views and framed/Sobel copies as extra rows (~730 at
# step 8); past 668 rows VGG's block-1 maps exceed 2^31 elements, beyond some CUDA kernels' 32-bit indexing (an
# illegal memory access on step 8's first batch, 2026-10-30).
CHUNK = 128
# Stored for the backward pass, those rows took ~105 MB each in bf16 and overflowed an 80 GB GPU (2026-10-31). Blocks
# 1-2 (features[:10], conv1_1 ... pool2, with the step-3 LRNs) hold most of that memory and ~30% of the FLOPs, so in
# training they are recomputed in the backward pass, one chunk at a time, instead of stored.
RECOMPUTED = 10


class ChunkedSequential(torch.nn.Sequential):
    """A Sequential that runs batches larger than CHUNK rows in chunks and, with gradients on, recomputes its first
    `recompute` modules in the backward pass instead of storing their activations. Identical results (VGG-16 has no
    batch norm, LRN is per image, features has no dropout); module names and state-dict keys are unchanged. Slices
    (VGGTaps' extra passes) recompute nothing."""
    recompute = 0

    def forward(self, x):
        if len(x) <= CHUNK:
            return self.rows(x)
        return torch.cat([self.rows(c) for c in x.split(CHUNK)])

    def rows(self, x):
        k = self.recompute if torch.is_grad_enabled() else 0
        if k:
            x = torch.utils.checkpoint.checkpoint(torch.nn.Sequential.forward, self[:k], x, use_reentrant=False)
        return torch.nn.Sequential.forward(self[k:], x)


class CentredL2(torch.nn.Module):
    """Per-image (l - mean(l)) / ||l - mean(l)||_2 over the last dim. No parameters."""

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class CentralZoom(torch.nn.Module):
    """Keep the central LABEL_DEGREES / VISUAL_DEGREES of the image (196 of 224 px) and resize it back
    (antialiased bilinear). Resizing commutes with the normalization, so this equals zooming the raw image."""

    def forward(self, x):
        h, w = x.shape[-2:]
        ch, cw = round(h * LABEL_DEGREES / VISUAL_DEGREES), round(w * LABEL_DEGREES / VISUAL_DEGREES)
        top, left = (h - ch) // 2, (w - cw) // 2
        x = x[..., top:top + ch, left:left + cw]
        return torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=True)


class BinomialBlur(torch.nn.Module):
    """Fixed 3x3 binomial ([1,2,1] outer [1,2,1] / 16) depthwise blur with stride 2 and zero padding 1. No parameters;
    the kernel is a non-persistent buffer, so the state dict is unchanged."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', (torch.outer(k, k) / 16)[None, None], persistent=False)

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.kernel.expand(x.shape[1], 1, 3, 3), stride=2, padding=1,
                                          groups=x.shape[1])


class VGGTaps(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.zoom = CentralZoom()
        self.blur = BinomialBlur()
        self.v1_avgpool = torch.nn.AvgPool2d(2, 2)       # step 10: read by its hook only
        self.logits_centred_l2 = CentredL2()             # step 5: read by its hook only
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def run(self, x, start, stop):
        """features[start:stop] without calling the tap modules (their ops applied functionally)."""
        for i in range(start, stop):
            kind = TAPPED.get(i)
            x = (torch.nn.functional.max_pool2d(x, 2, 2) if kind == 'pool' else torch.relu(x) if kind == 'relu'
                 else self.features[i](x))
        return x

    def front(self, x):
        """Anti-aliased blocks 1-3 (steps 11-13): pool1 and pool2 as a stride-1 max then a stride-2 blur. Returns the
        conv3_3 ReLU output (pool3's input), 256x56x56."""
        x = self.blur(torch.nn.functional.max_pool2d(self.run(x, 0, 4), 2, 1))
        x = self.blur(torch.nn.functional.max_pool2d(self.run(x, 5, 9), 2, 1))
        return self.run(x, 10, 16)

    def forward(self, x):
        f, c = self.features, self.classifier
        if STEP >= 9:
            zoomed = c[:6](torch.flatten(self.avgpool(f(self.zoom(x))), 1))
        # the unmodified pass after the zoomed one, so the features.* and classifier taps hold the 8 deg image
        conv33 = f[:16](x)
        hidden = c[:6](torch.flatten(self.avgpool(f[16:](conv33)), 1))
        if STEP == 10:
            self.v1_avgpool(conv33)
        if STEP >= 11:
            b3 = self.front(x)
            self.v1_avgpool(b3)
            if STEP >= 12:
                f[16](b3)                                                         # V2: pool3 of the anti-aliased pass
            if STEP >= 13:
                f[18](f[17](torch.nn.functional.max_pool2d(b3, 2, 2)))           # V4: conv4_1 ReLU of that pass
        if STEP >= 14:
            # IT: pool4 averaged over four copies translated by (0,0), (0,2), (2,0), (2,2) px (reflect-pad the leading
            # edges, crop back). features.23 (2x2 stride-2 max pool) is called last on a map that holds the mean at
            # each window's top-left and -inf elsewhere, so it outputs exactly the mean and its hook holds it.
            h, w = x.shape[-2:]
            s = torch.cat([torch.nn.functional.pad(x, (dx, 0, dy, 0), mode='reflect')[..., :h, :w]
                           for dy, dx in ((0, 0), (0, 2), (2, 0), (2, 2))])
            it = torch.nn.functional.max_pool2d(self.run(s, 0, 23), 2, 2).unflatten(0, (4, -1)).mean(0)
            grid = torch.full((*it.shape[:2], 2 * it.shape[2], 2 * it.shape[3]), float('-inf'), dtype=it.dtype,
                              device=it.device)
            grid[..., ::2, ::2] = it
            f[23](grid)
        last = c[6]
        if STEP >= 5:
            self.logits_centred_l2(torch.nn.functional.linear(hidden, last.weight, last.bias))
        # classifier.6 is linear, so on the mean of the two fc7 vectors it gives the mean of the two views' logits
        return last((hidden + zoomed) / 2) if STEP >= 9 else last(hidden)


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    model = torchvision.models.vgg16(weights=None)
    if STEP >= 3:   # LRN after the ReLU before pool1 and pool2; Sequential(ReLU, LRN) in place keeps names and keys
        for i in (3, 8):
            model.features[i] = torch.nn.Sequential(model.features[i],
                                                    torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
    model.features = ChunkedSequential(*model.features)
    model.features.recompute = RECOMPUTED
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = VGGTaps(model).eval()
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
