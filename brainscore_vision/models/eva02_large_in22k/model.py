"""
Iter-6 record attempt — EVA-02 Large (MIM-pretrained ViT, merged-38M + IN22k/IN1k fine-tune, 448px).

Diversity bet #2. EVA-02 is masked-image-modeling pretrained (reconstructs CLIP features) then ImageNet
fine-tuned -> a self-supervised-style ViT with one of the strongest transfer/representation profiles, and
it ships a real 1000-logit head (label behavior works). MIM ViTs have historically scored well on
Brain-Score. Distinct training signal from DINOv2 (contrastive self-distillation) and ConvNeXt-V2 (FCMAE),
so it's an independent shot at the neural+behavior frontier. ViT-L/14, depth 24, 1024-dim, 448px (32x32
patches). timm public weights, CI-downloadable.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'eva02_large_in22k'
TIMM_NAME = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
IMAGE_SIZE = 448


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
    # 6 transformer blocks spread across depth-24
    return ['blocks.3', 'blocks.7', 'blocks.11', 'blocks.15', 'blocks.19', 'blocks.23']


def get_bibtex(model_identifier):
    return """@article{fang2023eva02,
      title={EVA-02: A Visual Representation for Neon Genesis},
      author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
      journal={arXiv:2303.11331}, year={2023}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
