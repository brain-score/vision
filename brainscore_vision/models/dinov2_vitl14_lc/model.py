"""
Iter-4 candidate — DINOv2 ViT-L/14 WITH the official ImageNet linear classifier (dinov2_vitl14_lc).

Why this exists: the headless DINOv2 (num_classes=0) cannot do Brain-Score's label-based behavior
(Geirhos/Baker route through LabelBehavior, which asserts the model emits 1000 ImageNet logits) —
so plain DINOv2 failed all those behavioral benchmarks. The official linear-classifier head restores
1000-logit outputs (forward(x) -> (B,1000)) while keeping DINOv2's self-supervised feature geometry,
which is best-in-sweep on the unfitted-RDM bottleneck (Coggan IT-rdm 0.63 vs CLIP iter-1's 0.055).

This is the "best of both" iter-4 thesis: self-supervised geometry (neural RDM) + a trained ImageNet
head (label behavior). Loaded via torch.hub; TORCH_HOME is redirected to scratch in the sbatch.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'dinov2_vitl14_lc'
IMAGE_SIZE = 224  # 224/14 = 16 patches/side


def get_model(name):
    assert name == IDENTIFIER
    # backbone (DINOv2 ViT-L/14, self-supervised) + linear_head (ImageNet linear probe). forward -> 1000 logits.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc', pretrained=True, trust_repo=True)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    # transformer blocks across depth for per-region neural commitment
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
