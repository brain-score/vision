"""alexnet_altahan_08: AlexNet fine-tuned on ImageNet with L2 adversarial training, at a 10 degree field of view, with
V1 read from a multi-scale, multi-phase and foveally warped population of conv1 (and conv2) responses.

torchvision AlexNet fine-tuned from the ImageNet weights on ImageNet-1k (SGD, lr 1e-3, momentum 0.9, weight decay 5e-4,
batch 256, 5.12M images; random-resized-crop and horizontal flip) with L2 PGD adversarial training on every batch
(eps 0.5 in [0, 1] pixel space, 3 steps of 2.5 * eps / 3, no random start). The architecture and state dict are
torchvision's (61,100,840 parameters). Training used the plain network; the readout below is fixed, parameter-free and
used only at test time.

Readouts (nothing is learned):
- V1 `v1_concat`: one population of flattened conv1 outputs (features.2, after ReLU and max pool) computed on several
  fixed transforms of the input, plus conv2 (features.5): the native image; the image bilinearly downsampled 2x and
  4x; upsampled sqrt(2); shifted 1 px left and 1 px up (replicate padding); radially warped about its centre (source
  radius r_s = R * (r / R) ** 2 for output radius r, R the half-width: magnified centrally, compressed peripherally);
  and the warped image downsampled 2x. Each block is scaled by a fixed constant, 1 / sqrt(its total variance over
  2000 ImageNet training images), and all but native conv1 by a further 1 / sqrt(2). 335,424 units.
- V2 and V4 features.7 (conv3 ReLU), IT features.12 (pool5), behaviour classifier.5 (fc7 ReLU): the plain network.
- Label logits: forward() is the plain network's; the readout passes do not feed it. The tap is registered before the
  base network's modules, so Brain-Score's default `logits` layer is still classifier.6, the output of forward().
"""
import functools

import torch
import torchvision

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'alexnet_altahan_08'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/alexnet_altahan_08.pt'
WEIGHTS_VERSION_ID = '_Vk9iZGPbSIqU38wdBViVISwWuR0.iNL'
WEIGHTS_SHA1 = '94614b8c341ef4af37ca9cd6ba35ae856405a211'   # checked after download (load_file does not verify it)

BIBTEX = """@incollection{NIPS2012_4824,
                  title = {ImageNet Classification with Deep Convolutional Neural Networks},
                  author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                  booktitle = {Advances in Neural Information Processing Systems 25},
                  pages = {1097--1105},
                  year = {2012},
                  publisher = {Curran Associates, Inc.}
                  }"""

IMAGE_SIZE = 224
LAYERS = ['v1_concat', 'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5']
# V1 reads v1_concat: flattened features.2 (64x27x27), features.5 (192x13x13), two coarse-scale features.2 blocks
# (conv1 on the input bilinearly downsampled 2x, 64x13x13, and 4x, 64x6x6) and one fine-scale features.2 block (conv1
# on the input bilinearly upsampled sqrt(2), 64x38x38) and one shifted features.2 block (conv1 on an unflipped copy
# of the input shifted 1 px left, replicate-padded on the right, 64x27x27: a +1 px stride-grid offset in width) and
# one v-shifted features.2 block (conv1 on an unflipped copy of the input shifted 1 px up, replicate-padded at the
# bottom, 64x27x27: a +1 px stride-grid offset in height) side by side (readout only), each block scaled by a fixed
# 1/sqrt(its total variance), with features.5 and the coarse, fine and both shifted conv1 blocks each further scaled
# by 1/sqrt(2).
# An eighth block, foveal-warped features.2 (conv1 on a copy of the input resampled with a fixed radial warp about the
# image centre, source radius r_s = R*(r/R)^2.0, R = half-width: magnified centrally, compressed peripherally,
# 64x27x27), is appended with the same treatment (1/sqrt(its total variance), then 1/sqrt(2)).
# A ninth block, warped-coarse features.2 (conv1 on the same exponent-2.0 foveal-warped input, then halved exactly 2x
# by bilinear interpolation, 64x13x13: eccentricity-varying scale at the coarse octave), gets the same treatment.
REGION_LAYER_MAP = {'V1': 'v1_concat', 'V2': 'features.7', 'V4': 'features.7', 'IT': 'features.12'}
BEHAVIORAL_READOUT_LAYER = 'classifier.5'
VISUAL_DEGREES = 10           # how large the model's field of view is taken to be (Brain-Score's default is 8)


# 1/sqrt(total variance) of each block: the sum over units of each unit's variance across 2000 ImageNet-train
# center crops (224/256, fixed random indices, seed 0, ddof=1), measured offline with these weights.
# Fixed constants: nothing is fit at scoring time.
# features.5 also gets an extra fixed factor 1/sqrt(2), halving its total variance relative to features.2, so
# conv2 carries 1/3 of the concatenated readout's variance instead of 1/2.
# Both coarse-scale conv1 blocks, the fine-scale one and both shifted ones get the same extra 1/sqrt(2), so the
# variance shares are 2:1:1:1:1:1:1; the foveal-warped block gets it too, making them 2:1:1:1:1:1:1:1, and so
# does the warped-coarse block: 2:1:1:1:1:1:1:1:1.
CONV2_SHARE_FACTOR = 2 ** -0.5
V1_BLOCK_NAMES = ('features.2', 'features.5', 'features.2@2x', 'features.2@4x', 'features.2@sqrt2',
                  'features.2@shift', 'features.2@vshift', 'features.2@fovea',
                  'features.2@fovea2x')  # v1_concat inputs
SHIFT_SCALE = 0.0016952560522252927  # 1/sqrt(total variance) of shifted features.2, measured as above
VSHIFT_SCALE = 0.0016950786089473097  # 1/sqrt(total variance) of v-shifted features.2, measured as above
FOVEA_SCALE = 0.0017763790335104728  # 1/sqrt(total variance) of foveal-warped features.2, measured as above
FOVEA2X_SCALE = 0.003274765201999184  # 1/sqrt(total variance) of warped-then-2x-halved features.2, measured as above
FOVEA_EXPONENT = 2.0  # source radius r_s = R*(r/R)**FOVEA_EXPONENT for output radius r, R = half-width
V1_BLOCK_SCALES = (0.0016949599831596123,                         # features.2: total variance 3.481e5
                   0.001020848531791071 * CONV2_SHARE_FACTOR,     # features.5: total variance 9.596e5
                   0.0031994059500200164 * CONV2_SHARE_FACTOR,    # 2x-coarse features.2: total variance 9.769e4
                   0.0064826031245902626 * CONV2_SHARE_FACTOR,    # 4x-coarse features.2: total variance 2.380e4
                   0.0013741135661333847 * CONV2_SHARE_FACTOR,    # sqrt(2)-fine features.2: total variance 5.296e5
                   SHIFT_SCALE * CONV2_SHARE_FACTOR,              # shifted features.2: total variance 3.480e5
                   VSHIFT_SCALE * CONV2_SHARE_FACTOR,             # v-shifted features.2: total variance 3.480e5
                   FOVEA_SCALE * CONV2_SHARE_FACTOR,              # foveal-warped features.2: total variance 3.169e5
                   FOVEA2X_SCALE * CONV2_SHARE_FACTOR)            # warped-coarse features.2: total variance 9.325e4


def foveal_grid(height, width, device, dtype):
    """Fixed grid_sample grid (1, H, W, 2) for the radial warp about the image centre: an output pixel at normalised
    radius r (R = 1 at the half-width) samples the input at radius r**FOVEA_EXPONENT along the same direction. Pixel
    centres in align_corners=False coordinates; corners (r > 1) fall outside and take the border."""
    ys = (torch.arange(height, device=device, dtype=dtype) * 2 + 1) / height - 1
    xs = (torch.arange(width, device=device, dtype=dtype) * 2 + 1) / width - 1
    y, x = torch.meshgrid(ys, xs, indexing='ij')
    gain = torch.sqrt(x * x + y * y) ** (FOVEA_EXPONENT - 1)  # r_s / r
    return torch.stack((x * gain, y * gain), dim=-1).unsqueeze(0)


class FlattenConcat(torch.nn.Module):
    """Parameter-free: flattens each input to (batch, -1), multiplies it by a fixed constant, and concatenates
    them along the unit axis."""

    def __init__(self, scales):
        super().__init__()
        self.scales = tuple(scales)  # plain floats, not buffers: the state dict is unchanged

    def forward(self, *xs):
        return torch.cat([torch.flatten(x, 1) * c for x, c in zip(xs, self.scales)], dim=1)


class ConcatV1AlexNet(torch.nn.Module):
    """AlexNet with a parameter-free tap: 'v1_concat' joins the flattened features.2 (conv1, 64x27x27),
    features.5 (conv2, 192x13x13), coarse-scale features.2 (conv1 on the input bilinearly downsampled 2x,
    64x13x13, and 4x, 64x6x6), fine-scale features.2 (conv1 on the input bilinearly upsampled sqrt(2), 64x38x38) and
    shifted features.2 (conv1 on the input shifted 1 px left, replicate-padded, 64x27x27) and v-shifted features.2
    (conv1 on the input shifted 1 px up, replicate-padded, 64x27x27) and foveal-warped features.2 (conv1 on the input
    radially resampled about its centre, 64x27x27) and warped-coarse features.2 (conv1 on the same warped input halved
    2x, 64x13x13) outputs, scaled so their total variances are 2:1:1:1:1:1:1:1:1, into one population for the V1 readout. The main path is unchanged, so the logits are the base network's. The tap is registered first,
    so the base network's module names and state dict are unchanged and its classifier stays the last-registered
    module."""

    def __init__(self, base):
        super().__init__()
        self.v1_concat = FlattenConcat(V1_BLOCK_SCALES)
        self.features, self.avgpool, self.classifier = base.features, base.avgpool, base.classifier

    def forward(self, x):
        x_in = x
        conv1 = self.features[:3](x)
        x = self.features[3:6](conv1)
        # Coarse scales: the same conv1 stack on 2x- and 4x-downsampled inputs. Called via .forward() so that hooks on
        # features.0-2 (activation extraction) only ever see the full-resolution pass.
        # The 4x scale halves the 2x input again, so each octave is box-filtered from the one above (no aliasing).
        half = torch.nn.functional.interpolate(x_in, scale_factor=0.5, mode='bilinear', align_corners=False)
        quarter = torch.nn.functional.interpolate(half, scale_factor=0.5, mode='bilinear', align_corners=False)
        # Fine scale: the same conv1 stack on the input bilinearly upsampled sqrt(2) (224 -> 316 px).
        fine = torch.nn.functional.interpolate(x_in, scale_factor=2 ** 0.5, mode='bilinear', align_corners=False)
        # Shifted: the same conv1 stack on the unflipped input shifted 1 px left (replicate-pad the right edge, crop the
        # left column), so its stride grid sits +1 px in width from the native one (conv1 centres at 4i+4 instead of
        # 4i+3), with the native filters.
        shift = torch.nn.functional.pad(x_in, (0, 1, 0, 0), mode='replicate')[..., 1:]
        # V-shifted: the same, 1 px up in height (replicate-pad the bottom row, crop the top row), so its stride grid
        # sits +1 px in height from the native one (conv1 centres at rows 4i+4 instead of 4i+3), 0 in width.
        vshift = torch.nn.functional.pad(x_in, (0, 0, 0, 1), mode='replicate')[..., 1:, :]
        # Foveal warp: the same conv1 stack on the input resampled with a fixed radial warp (source radius r**2.0 for
        # output radius r, normalised to the half-width), bilinear with border padding, so scale grows towards the centre.
        grid = foveal_grid(x_in.shape[-2], x_in.shape[-1], x_in.device, x_in.dtype).expand(len(x_in), -1, -1, -1)
        fovea = torch.nn.functional.grid_sample(x_in, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # Warped coarse: the warped input halved exactly 2x (bilinear at scale 0.5 = 2x2 box average), as for the 2x block.
        fovea_half = torch.nn.functional.interpolate(fovea, scale_factor=0.5, mode='bilinear', align_corners=False)
        coarse, coarser = half, quarter
        for layer in self.features[:3]:
            coarse, coarser, fine = layer.forward(coarse), layer.forward(coarser), layer.forward(fine)
            shift, vshift, fovea = layer.forward(shift), layer.forward(vshift), layer.forward(fovea)
            fovea_half = layer.forward(fovea_half)
        self.v1_concat(conv1, x, coarse, coarser, fine, shift, vshift, fovea, fovea_half)  # read-out only; its output does not feed the rest of the network
        x = self.avgpool(self.features[6:](x))
        return self.classifier(torch.flatten(x, 1))


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
    model = ConcatV1AlexNet(model).eval()
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
