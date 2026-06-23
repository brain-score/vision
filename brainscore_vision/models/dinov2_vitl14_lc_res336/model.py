"""
Iter-7 record attempt — DINOv2 ViT-L/14 + ImageNet linear classifier at HIGHER RESOLUTION (336px).

Same model + recipe as our best dinov2_vitl14_lc (overall 0.41), changing ONLY the input resolution
224 -> 336 (24x24 patches vs 16x16, ~2.25x more tokens). DINOv2 interpolates its positional embeddings,
so the linear_head still works. Rationale: finer spatial detail per visual degree -> richer high-spatial-
frequency content for V1/V2 and finer-grained representational-dissimilarity matrices for the fMRI/RDM
benchmarks (our neural weak spot). Isolated single-lever experiment (resolution) vs dinov2_vitl14_lc;
keeps the winning ViT-L size and the proven `_lc` head (forward -> 1000 logits, behavior works).
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'dinov2_vitl14_lc_res336'
IMAGE_SIZE = 336  # 336/14 = 24 patches/side


def get_model(name):
    assert name == IDENTIFIER
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc', pretrained=True, trust_repo=True)
    model.eval()
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
    return """@article{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision},
      author={Oquab, Maxime and others},
      journal={arXiv:2304.07193}, year={2023}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
