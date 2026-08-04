"""
Iter-9 CHAMPION CANDIDATE — DINOv2 ViT-L/14 + REGISTERS + feature-embedding behavioral readout.

Stacks the two behavior winners. (1) REGISTERS: dinov2_vitl14_reg_lc just hit 0.44 (new best), and the
gain was BEHAVIOR 0.46->0.54 (cleaner artifact-free features -> more human-aligned decisions) at ~flat
neural. (2) FEATURE READOUT: read the decoder-fit similarity behaviors (Hebart-match, Maniquet-confusion,
Rajalingham-i2n) off the rich 5120-d pre-classifier embedding instead of 1000-d logits. Both target the
behavior half (the live axis); resolution was dropped (336px HURT, 0.41->0.33). Same `_lc` recipe, 224px;
label behaviors (Geirhos/Baker) still use forward(x)->1000 logits.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
import torch.nn as nn
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'dinov2_vitl14_reg_lc_featread'
IMAGE_SIZE = 224


class _LCFeatureReadout(nn.Module):
    """DINOv2 layers=4 linear-classifier forward, exposing the pre-head 5120-d embedding as `feature`."""
    def __init__(self, lc):
        super().__init__()
        self.backbone = lc.backbone
        self.linear_head = lc.linear_head
        self.feature = nn.Identity()  # behavioral readout hook target -> (B, 5*embed_dim)

    def forward(self, x):
        xs = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input = torch.cat([xs[0][1], xs[1][1], xs[2][1], xs[3][1], xs[3][0].mean(dim=1)], dim=1)
        feat = self.feature(linear_input)
        return self.linear_head(feat)


def get_model(name):
    assert name == IDENTIFIER
    lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc', pretrained=True, trust_repo=True)
    model = _LCFeatureReadout(lc).eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return [
        'backbone.blocks.3', 'backbone.blocks.7', 'backbone.blocks.11',
        'backbone.blocks.15', 'backbone.blocks.19', 'backbone.blocks.23',
    ]


def get_bibtex(model_identifier):
    return """@article{darcet2023registers, title={Vision Transformers Need Registers},
      author={Darcet, Timothee and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
      journal={arXiv:2309.16588}, year={2023}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
