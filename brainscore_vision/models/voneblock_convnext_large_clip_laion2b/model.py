"""
Brain-Score vision plugin — Iteration 2.

Hybrid: a FIXED (untrained, deterministic) VOneBlock V1 front-end run IN PARALLEL with the
full pretrained ConvNeXt-Large (LAION-2B CLIP, in12k/in1k, 384px) from iteration 1.

Why parallel instead of replacing the stem: Brain-Score commits a *different layer per
region*. Iteration 1 showed this backbone is strong on V4/IT/behavior but weak on V1/V2
(0.22/0.25) because its early layers are not Gabor-like. By exposing a biologically-tuned
VOneBlock as an additional candidate layer, the framework's per-region LayerSelection can
pick the VOneBlock for V1/V2 while keeping ConvNeXt layers for V4/IT and the ConvNeXt head
for behavior. No training, no loss of the backbone's behavioral strength.

The two streams want different input ranges, so the wrapper feeds [0,1] (good for the linear
Gabor front-end + PLS readout) and applies ConvNeXt's own normalization internally.
"""
import functools
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from .vonenet.vonenet.vonenet import VOneNet

IDENTIFIER = 'voneblock_convnext_large_clip_laion2b'
TIMM_NAME = 'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384'
IMAGE_SIZE = 384
VISUAL_DEGREES = 8
# VOneBlock runs at its canonical 224px (28 ppd) via internal downsampling. At 384px its
# (512,96,96)=4.7M-feature map overflows int32 in the PCA's LAPACK SVD (1000*4.7M>2.1e9);
# at 224px it is (512,56,56)=1.6M -> safely under the limit, and 28 ppd is its tuned resolution.
VONE_SIZE = 224


class VOneBlockConvNeXt(nn.Module):
    """Fixed VOneBlock V1 front-end (parallel) + pretrained ConvNeXt backbone."""

    def __init__(self, timm_name=TIMM_NAME, image_size=IMAGE_SIZE, visual_degrees=VISUAL_DEGREES):
        super().__init__()
        # Standalone, fixed, deterministic VOneBlock (noise_mode=None -> no stochasticity).
        # Gabor parameters are tuned to primate V1 and scaled to the input's pixels-per-degree.
        self.voneblock = VOneNet(model_arch=None, image_size=VONE_SIZE, visual_degrees=visual_degrees,
                                 noise_mode=None, simple_channels=256, complex_channels=256,
                                 sf_max=9, sf_min=0, k_exc=25, ksize=25, stride=4, gabor_seed=0)
        # Pretrained backbone (strong V4/IT/behavior from iteration 1).
        self.convnext = timm.create_model(timm_name, pretrained=True)
        cfg = self.convnext.default_cfg
        mean = torch.tensor(cfg['mean']).view(1, 3, 1, 1)
        std = torch.tensor(cfg['std']).view(1, 3, 1, 1)
        self.register_buffer('cn_mean', mean)
        self.register_buffer('cn_std', std)

    def forward(self, x):  # x in [0,1] at IMAGE_SIZE (preprocessing uses mean=0,std=1)
        # VOneBlock at its canonical 224px; activations captured by the 'voneblock' hook (V1/V2 candidate).
        x_vone = F.interpolate(x, size=VONE_SIZE, mode='bilinear', align_corners=False)
        _ = self.voneblock(x_vone)
        # ConvNeXt at full IMAGE_SIZE with its own (CLIP) normalization, applied internally.
        x_cn = (x - self.cn_mean) / self.cn_std
        return self.convnext(x_cn)


def get_model(name):
    assert name == IDENTIFIER
    model = VOneBlockConvNeXt()
    model.eval()
    # Feed [0,1] (no ImageNet normalization here; ConvNeXt is normalized inside forward).
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE,
                                       normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1))
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return [
        'voneblock',                                  # biologically-tuned V1 -> V1/V2 candidate
        'convnext.stem',
        'convnext.stages.0.blocks.2',
        'convnext.stages.1.blocks.2',
        'convnext.stages.2.blocks.0', 'convnext.stages.2.blocks.8',
        'convnext.stages.2.blocks.17', 'convnext.stages.2.blocks.26',
        'convnext.stages.3.blocks.2',
        'convnext.head.global_pool',                  # high-level pooled feature -> behavior
    ]


def get_bibtex(model_identifier):
    return """@inproceedings{dapello2020simulating,
  title={Simulating a primary visual cortex at the front of CNNs improves robustness to image perturbations},
  author={Dapello, Joel and Marques, Tiago and Schrimpf, Martin and Geiger, Franziska and Cox, David D and DiCarlo, James J},
  booktitle={NeurIPS},
  year={2020}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
