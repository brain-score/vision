"""
Brain-Score submission: BEiT-3 Large, ImageNet-22k + ImageNet-1k fine-tuned, 224px.

Strategy notes: BEiT-3 is a masked-image-modeling + image-text multimodal backbone (one of the
strongest transfer/representation models), here with a real 1000-logit ImageNet head (label behavior
works) at 88.4% top-1. It brings a training signal distinct from both CLIP (contrastive alignment)
and DINOv2 (self-distillation): MIM-style pretraining has historically matched primate IT/behavior
representations well, while the strong supervised head should carry the engineering category
(ImageNet / ImageNet-C / ObjectNet). Auto layer selection (no fixed region map) lets Brain-Score's
LayerSelection cross-validate whole-block and attention sub-module candidates per region.
Behavioral readout = fc_norm (pooled pre-classifier embedding). Compute: ~61 GMACs @ 224px —
lighter than the current #1, so the full benchmark suite should complete without timeouts.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images

IDENTIFIER = 'beit3_large_patch16_224_in1k'
TIMM_NAME = 'beit3_large_patch16_224.in22k_ft_in1k'
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
    return [
        # whole-block outputs across depth
        'blocks.3', 'blocks.5', 'blocks.6', 'blocks.7', 'blocks.9', 'blocks.11',
        'blocks.13', 'blocks.15', 'blocks.17', 'blocks.19', 'blocks.21', 'blocks.23',
        # attention sub-module activations
        'blocks.6.attn', 'blocks.7.attn', 'blocks.10.attn', 'blocks.12.attn',
        # pooled embedding for behavioral readout
        'fc_norm',
    ]


def get_bibtex(model_identifier):
    return """@article{wang2023beit3,
  title={Image as a Foreign Language: BEiT-3 for Vision and Vision-Language Pretraining},
  author={Wang, Wenhui and Bao, Hangbo and Gao, Li and Li, Furu and others},
  journal={arXiv preprint arXiv:2208.10442},
  year={2022}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
