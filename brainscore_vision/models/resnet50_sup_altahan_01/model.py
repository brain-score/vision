"""resnet50_sup_altahan_01: ResNet-50 with local response normalisation in the stem and after layer1, fine-tuned on
ImageNet with a SimCLR auxiliary and framed, edge-map and wide-crop training views; V1, V2 and V4 read from an
anti-aliased pass, IT averaged over small image shifts, behaviour from centred, L2-normalised logits.

torchvision ResNet-50, initialised from its IMAGENET1K_V1 weights, with a parameter-free local response normalisation
(size 5, alpha 1e-4, beta 0.75, k 2) after the stem ReLU (before the max pool) and after layer1. Fine-tuned on
ImageNet-1k with batch norm frozen (running statistics kept, scale and shift trained): SGD, lr 1e-3, momentum 0.9,
weight decay 5e-4, batch 256, 5.12M images; cross-entropy plus a SimCLR NT-Xent auxiliary loss (weight 0.5,
temperature 0.2) on the 2048-d pooled features through a projection head that is discarded after training. Training
views: random-resized-crop with area scale 0.35-1 and horizontal flip; for the cross-entropy input, 10% of the images
are replaced by their Sobel edge map, and 80% are then shrunk by a factor U[0.35, 1] and pasted centred on a uniform
grey canvas of random level; the SimCLR views are unframed. The LRN layers have no weights, so the network has
ResNet-50's 25,557,032 parameters and the state dict is torchvision's.

Readouts, all parameter-free:
  V1      layer2's output from an anti-aliased front end (Zhang 2019, "Making convolutional networks shift-invariant
          again"): the stem max pool at stride 1 followed by a stride-2 3x3 binomial blur, and layer2.0's stride-2
          conv and shortcut at stride 1 followed by the blur; then a 2x2 average pool (512x14x14)
  V2, V4  layer3.0 of the same anti-aliased pass (1024x14x14)
  IT      layer4.0 averaged over four copies of the image translated by 0 or 2 px in each direction (2048x7x7)
  behaviour  the 1000 logits, centred and L2-normalised per image
  logits  (label benchmarks) the mean of the logits of the 8 degree image and of its central 7 degrees resized back
The extra passes run inside forward() before the plain pass, and a forward hook keeps a module's last output, so the
torchvision-named layers hold the plain pass of the 8 degree image. The readout modules are registered before the
network's own, so Brain-Score's default `logits` layer is fc, the output of forward(). The model code is shared with
variants that switch readouts on through STEP; at STEP = 14 all of them are on.
"""
import functools

import torch
import torch.nn.functional as F
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'resnet50_sup_altahan_01'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/resnet50_sup_altahan_01.pt'
WEIGHTS_VERSION_ID = 'qIsuqCIIQ8dsE9KGZ1YGYxkfXKlED6Kb'
WEIGHTS_SHA1 = '052f87c30e8b043e66aa574f6f7bb4c8f2778319'   # checked after download (load_file does not verify it)

BIBTEX = """@inproceedings{he2016deep,
                  title = {Deep Residual Learning for Image Recognition},
                  author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
                  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                  pages = {770--778},
                  year = {2016},
                  url = {https://arxiv.org/abs/1512.03385}
                  }"""

STEP = 14
IMAGE_SIZE = 224
REGION_LAYER_MAP = {'V1': 'v1_avgpool' if STEP >= 10 else 'layer2.3',
                    'V2': 'aa_layer3' if STEP >= 12 else 'layer3.0',
                    'V4': 'aa_layer3' if STEP >= 13 else 'layer3.0' if STEP >= 2 else 'layer2.3',
                    'IT': 'it_shift' if STEP >= 14 else 'layer4.0'}
BEHAVIORAL_READOUT_LAYER = 'logits_centred_l2' if STEP >= 5 else 'fc' if STEP >= 4 else 'avgpool'
LAYERS = list(dict.fromkeys(['conv1', 'layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
                             'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0',
                             'layer4.1', 'layer4.2', 'avgpool', *REGION_LAYER_MAP.values(), BEHAVIORAL_READOUT_LAYER]))
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)
LABEL_DEGREES = 7             # step 9: forward()'s logits average the VISUAL_DEGREES view with this magnified view


def lrn():
    return torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)


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
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=True)


class BinomialBlur(torch.nn.Module):
    """Fixed 3x3 binomial ([1,2,1] outer [1,2,1] / 16) depthwise blur with stride 2 and zero padding 1. No parameters;
    the kernel is a non-persistent buffer, so the state dict is unchanged."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', (torch.outer(k, k) / 16)[None, None], persistent=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel.to(x.dtype).expand(x.shape[1], 1, 3, 3), stride=2, padding=1, groups=x.shape[1])


class ResNetTaps(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.zoom = CentralZoom()
        self.blur = BinomialBlur()
        self.v1_avgpool = torch.nn.AvgPool2d(2, 2)       # step 10+: read by its hook only
        self.aa_layer3 = torch.nn.Identity()             # steps 12+ (V2), 13+ (V4): layer3.0 of the anti-aliased pass
        self.it_shift = torch.nn.Identity()              # step 14: layer4.0 averaged over four shifts
        self.logits_centred_l2 = CentredL2()             # step 5+: read by its hook only
        for name, child in base.named_children():       # conv1 ... fc, under torchvision's names
            setattr(self, name, child)

    def train(self, mode=True):
        """Batch norm stays in eval mode (frozen statistics); its scale and shift still train."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        return self

    def stem(self, x):
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))

    def pooled(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return torch.flatten(self.avgpool(x), 1)

    def aa_block(self, b, x):
        """A stride-2 bottleneck anti-aliased (steps 11-13): conv2 at stride 1, its BN and ReLU, then a stride-2 blur
        before conv3; the shortcut is blurred (stride 2) before its 1x1 conv at stride 1. Same weights."""
        out = b.relu(b.bn1(b.conv1(x)))
        out = self.blur(b.relu(b.bn2(F.conv2d(out, b.conv2.weight, stride=1, padding=1))))
        out = b.bn3(b.conv3(out))
        conv, bn = b.downsample
        return b.relu(out + bn(F.conv2d(self.blur(x), conv.weight)))

    def forward(self, x):
        if STEP >= 9:
            zoomed = self.pooled(self.zoom(x))
        if STEP >= 11:   # anti-aliased front end: stem max pool at stride 1 + blur, anti-aliased layer2.0
            s = self.blur(F.max_pool2d(self.relu(self.bn1(self.conv1(x))), 3, 1, 1))
            l2 = self.layer2[1:](self.aa_block(self.layer2[0], self.layer1(s)))
            self.v1_avgpool(l2)
            if STEP >= 12:
                self.aa_layer3(self.layer3[0](l2))
        if STEP >= 14:
            # IT: layer4.0 averaged over four copies translated by (0,0), (0,2), (2,0), (2,2) px (reflect-pad the
            # leading edges, crop back)
            h, w = x.shape[-2:]
            s = torch.cat([F.pad(x, (dx, 0, dy, 0), mode='reflect')[..., :h, :w]
                           for dy, dx in ((0, 0), (0, 2), (2, 0), (2, 2))])
            it = self.layer4[0](self.layer3(self.layer2(self.layer1(self.stem(s)))))
            self.it_shift(it.unflatten(0, (4, -1)).mean(0))
        # the unmodified pass last, so the torchvision-named taps hold the 8 deg image
        l2 = self.layer2(self.layer1(self.stem(x)))
        if STEP == 10:
            self.v1_avgpool(l2)
        hidden = torch.flatten(self.avgpool(self.layer4(self.layer3(l2))), 1)
        if STEP >= 5:
            self.logits_centred_l2(F.linear(hidden, self.fc.weight, self.fc.bias))
        # fc is linear, so on the mean of the two pooled vectors it gives the mean of the two views' logits
        return self.fc((hidden + zoomed) / 2) if STEP >= 9 else self.fc(hidden)


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    model = torchvision.models.resnet50(weights=None)
    if STEP >= 3:   # parameter-free LRN after the stem ReLU and after layer1; names and state-dict keys unchanged
        model.relu = torch.nn.Sequential(model.relu, lrn())
        model.layer1.append(lrn())
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = ResNetTaps(model).eval()
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
