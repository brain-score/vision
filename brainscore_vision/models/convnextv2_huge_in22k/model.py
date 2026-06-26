"""
Iter-6 record attempt — ConvNeXt-V2 Huge (FCMAE self-supervised + ImageNet-22k/1k fine-tune, 384px).

Thesis (diversify the inductive bias): our DINOv2 ViT wins HIGH-level geometry (IT-RDM) but is weak on
EARLY-visual RDMs (Coggan V1/V2 ~0.04-0.09) because ViTs lack conv/Gabor-like early features. ConvNeXt-V2
is (a) SELF-SUPERVISED pretrained (FCMAE masked-autoencoder -> brain-like representational geometry, like
DINOv2) AND (b) CONVOLUTIONAL (hierarchical local features -> strong V1/V2/V4) AND (c) ImageNet-fine-tuned
(real 1000-logit head -> label behavior works out of the box). So it attacks neural at ALL four regions
while keeping behavior, where DINOv2-ViT can't. Huge = 660M params, stage depths [3,3,27,3], dims to 2816.
timm public weights (hf_hub_id=timm/...), CI-downloadable. Mirrors our CI-validated convnext plugin layout.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'convnextv2_huge_in22k'
TIMM_NAME = 'convnextv2_huge.fcmae_ft_in22k_in1k_384'
IMAGE_SIZE = 384


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
    # same hierarchy spread as the CI-validated convnext_large plugin (huge shares [3,3,27,3] depths)
    return [
        'stem',
        'stages.0.blocks.2',
        'stages.1.blocks.2',
        'stages.2.blocks.0', 'stages.2.blocks.8', 'stages.2.blocks.17', 'stages.2.blocks.26',
        'stages.3.blocks.2',
        'head.global_pool',
    ]


def get_bibtex(model_identifier):
    return """@article{woo2023convnextv2,
      title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
      author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Kweon, In So and Xie, Saining},
      journal={CVPR}, year={2023}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
