"""
Iter-3 candidate B — SUPERVISED ConvNeXt-Large (non-CLIP control).

Same architecture as iter-1 (convnext_large) but ImageNet-22k supervised pretraining +
in1k fine-tune at 384px, instead of LAION-2B CLIP. This isolates whether CLIP's
language-aligned feature geometry is what tanks the unfitted-RDM benchmarks: if removing
CLIP alone lifts the RDM scores, the backbone choice is the lever.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'convnext_large_in22k_supervised'
TIMM_NAME = 'convnext_large.fb_in22k_ft_in1k_384'
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
    return [
        'stem',
        'stages.0.blocks.2',
        'stages.1.blocks.2',
        'stages.2.blocks.0', 'stages.2.blocks.8', 'stages.2.blocks.17', 'stages.2.blocks.26',
        'stages.3.blocks.2',
        'head.global_pool',
    ]


def get_bibtex(model_identifier):
    return """@article{liu2022convnet,
      title={A ConvNet for the 2020s},
      author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
      journal={CVPR}, year={2022}}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
