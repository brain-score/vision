"""
Iter-5 record attempt — DINOv2 ViT-g/14 WITH REGISTERS + ImageNet linear classifier (dinov2_vitg14_reg_lc).

Same scale-up as dinov2_vitg14_lc but with 4 register tokens. Registers absorb the high-norm artifact
tokens that otherwise pollute DINOv2's patch-token feature maps, yielding cleaner dense geometry. Our
weakest neural leaves are the early-visual RDMs (Coggan V1/V2/V4 ~0.04-0.09); cleaner patch tokens
directly target those. Register tokens are extra input tokens, NOT extra blocks, so block indexing
(depth 40) is identical to the non-reg variant. Same `_lc` recipe -> forward -> 1000 logits (behavior works).
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'dinov2_vitg14_reg_lc'
IMAGE_SIZE = 224  # 224/14 = 16 patches/side


def get_model(name):
    assert name == IDENTIFIER
    # backbone (DINOv2 ViT-g/14 + 4 registers) + linear_head (ImageNet linear probe). forward -> 1000 logits.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc', pretrained=True, trust_repo=True)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return [
        'backbone.blocks.5', 'backbone.blocks.12', 'backbone.blocks.18',
        'backbone.blocks.25', 'backbone.blocks.32', 'backbone.blocks.39',
    ]


def get_bibtex(model_identifier):
    return """@article{darcet2023registers,
      title={Vision Transformers Need Registers},
      author={Darcet, Timoth\\'ee and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
      journal={arXiv:2309.16588}, year={2023}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
