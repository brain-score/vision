"""
Iter-10 — ConvNeXt-XXLarge CLIP (LAION-2B) fine-tuned on ImageNet-1k (convnext_xxlarge.clip_laion2b_soup_ft_in1k).

Biggest CLIP-pretrained ConvNeXt. The board shows CLIP+ft and supervised ConvNeXts both near the top, and for
these (unlike DINOv2) SCALING HELPS (convnext_xlarge 0.45 > large 0.44). This is the largest CLIP ConvNeXt:
conv inductive bias (early-visual V1/V2) + CLIP human-aligned features + ImageNet head. 256px native.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools, timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images

IDENTIFIER = 'convnext_xxlarge_clip'
TIMM_NAME = 'convnext_xxlarge.clip_laion2b_soup_ft_in1k'
IMAGE_SIZE = 256


def get_model(name):
    assert name == IDENTIFIER
    model = timm.create_model(TIMM_NAME, pretrained=True).eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return ['stem', 'stages.0.blocks.2', 'stages.1.blocks.3',
            'stages.2.blocks.0', 'stages.2.blocks.10', 'stages.2.blocks.20', 'stages.2.blocks.29',
            'stages.3.blocks.2', 'head.global_pool']


def get_bibtex(model_identifier):
    return "@article{woo2023convnextv2, title={ConvNeXt}, author={Liu et al}, year={2022}}"


if __name__ == '__main__':
    check_models.check_base_models(__name__)
