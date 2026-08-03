"""
Iter-7 record attempt — DINOv2 ViT-L/14 WITH REGISTERS + ImageNet linear classifier (dinov2_vitl14_reg_lc).

Data-driven: our best model is dinov2_vitl14_lc (overall 0.41, neural 0.36). Scaling to ViT-g LOWERED
neural (0.28) and V1 (0.27) -> ViT-L/14 is the brain-alignment sweet spot; do NOT scale. The remaining
neural headroom is the EARLY-visual / dense fMRI-RDM benchmarks (Coggan V1/V2 ~0.04-0.09). DINOv2 emits
high-norm ARTIFACT patch tokens that pollute dense feature maps; REGISTERS (Darcet 2023) absorb them ->
cleaner patch-token geometry -> directly targets those weak dense RDM/fMRI leaves. Keep the WINNING size
(ViT-L), keep the proven `_lc` recipe (backbone + linear_head, forward -> 1000 logits, behavior works);
change ONLY the artifact cleanup. Isolated single-lever experiment vs dinov2_vitl14_lc.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'dinov2_vitl14_reg_lc'
IMAGE_SIZE = 224  # 224/14 = 16 patches/side (match dinov2_vitl14_lc for a clean registers-only comparison)


def get_model(name):
    assert name == IDENTIFIER
    # ViT-L/14 + 4 register tokens, with the official ImageNet linear classifier head.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc', pretrained=True, trust_repo=True)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    # identical depth-24 spread as the proven dinov2_vitl14_lc (registers add tokens, not blocks)
    return [
        'backbone.blocks.3', 'backbone.blocks.7', 'backbone.blocks.11',
        'backbone.blocks.15', 'backbone.blocks.19', 'backbone.blocks.23',
    ]


def get_bibtex(model_identifier):
    return """@article{darcet2023registers,
      title={Vision Transformers Need Registers},
      author={Darcet, Timoth\\'ee and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
      journal={arXiv:2309.16588}, year={2023}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
