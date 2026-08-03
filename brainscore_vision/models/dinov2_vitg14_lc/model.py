"""
Iter-5 record attempt — DINOv2 ViT-g/14 WITH the official ImageNet linear classifier (dinov2_vitg14_lc).

Scale-up of our current #1 (dinov2_vitl14_lc, site overall 0.41). The lead over the ConvNeXts came
ENTIRELY from the neural side (0.36 vs 0.20) — DINOv2's self-supervised representational geometry is
best-in-sweep on the RDM/fMRI benchmarks that now dominate the post-Feb-2026 leaderboard. ViT-g/14
(1.1B params, depth 40, embed 1536) is the largest DINOv2 backbone; richer geometry should lift neural,
and a stronger linear probe should lift shape-bias behavior. Same torch.hub `_lc` recipe (backbone +
linear_head, forward -> 1000 logits) as the proven #1, so label-behavior (Geirhos/Baker) works.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'dinov2_vitg14_lc'
IMAGE_SIZE = 224  # 224/14 = 16 patches/side


def get_model(name):
    assert name == IDENTIFIER
    # backbone (DINOv2 ViT-g/14, self-supervised) + linear_head (ImageNet linear probe). forward -> 1000 logits.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc', pretrained=True, trust_repo=True)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    # 6 transformer blocks spread across depth-40 (mirrors the L/14 commitment at the same depth-fractions)
    return [
        'backbone.blocks.5', 'backbone.blocks.12', 'backbone.blocks.18',
        'backbone.blocks.25', 'backbone.blocks.32', 'backbone.blocks.39',
    ]


def get_bibtex(model_identifier):
    return """@article{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision},
      author={Oquab, Maxime and others},
      journal={arXiv:2304.07193}, year={2023}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
