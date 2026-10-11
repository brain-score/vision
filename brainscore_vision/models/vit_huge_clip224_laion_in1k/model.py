"""
Brain-Score submission: ViT-H/14 LAION-2B CLIP, pure ImageNet-1k fine-tune, 224px.

Strategy notes: the leaderboard's top models are CLIP-pretrained + ImageNet-FINE-TUNED backbones.
For the OpenAI ViT-L/14 family, the pure in1k fine-tune beat the in12k+in1k variant by a wide margin
(0.48 vs 0.42). The board's ViT-H entry is only the in12k_in1k variant (0.44); this submission tests
the same in1k-only uplift at ViT-H scale (632M params, 88.0% IN top-1, native 224px so no resolution
mismatch). Region-layer commitments follow the proven ViT-H-tuned sub-module map (early-mid attention
sublayers for V1/V2/V4, ~1/3-depth attention for IT). Behavioral readout = fc_norm, the pooled
pre-classifier embedding, which is the strongest similarity space for the odd-one-out / confusion
behavioral benchmarks. Compute (~167 GMACs @224) is within the fully-scored envelope.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images

IDENTIFIER = 'vit_huge_clip224_laion_in1k'
TIMM_NAME = 'vit_huge_patch14_clip_224.laion2b_ft_in1k'
IMAGE_SIZE = 224


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    model = timm.create_model(TIMM_NAME, pretrained=True)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return ['blocks.11.attn', 'blocks.7.attn.qkv', 'blocks.7.attn', 'blocks.6.attn.qkv', 'fc_norm']


def get_bibtex(model_identifier):
    return """@article{cherti2023reproducible,
  title={Reproducible Scaling Laws for Contrastive Language-Image Learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  journal={arXiv preprint arXiv:2212.07173},
  year={2022}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
