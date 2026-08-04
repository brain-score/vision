"""
Brain-Score vision plugin — Iteration 1 baseline.

ConvNeXt-Large with LAION-2B CLIP pretraining (in12k+in1k fine-tune, 384px), exposed
with FINE-GRAINED block-level candidate layers so the framework's per-region
LayerSelection has good options (existing convnext entries only expose the 4 coarse
stage outputs). region_layer_map is intentionally left unset in __init__.py so the
framework auto-commits the best layer per region.

Rationale: strong large-scale pretraining + high resolution -> strong behavioral
alignment (Geirhos2021 / Rajalingham2018), which dominates the overall leaderboard rank.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

IDENTIFIER = 'convnext_large_clip_laion2b_finelayers'
# LAION-2B CLIP-pretrained ConvNeXt-Large, in12k+in1k fine-tuned at 384px.
TIMM_NAME = 'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384'
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
    # ConvNeXt-Large depths = (3, 3, 27, 3). Fine-grained taps across the hierarchy:
    return [
        'stem',
        'stages.0.blocks.2',
        'stages.1.blocks.2',
        'stages.2.blocks.0', 'stages.2.blocks.8', 'stages.2.blocks.17', 'stages.2.blocks.26',
        'stages.3.blocks.2',
        'head.global_pool',   # high-level pooled feature -> behavioral readout
    ]


def get_bibtex(model_identifier):
    return """@article{liu2022convnet,
      title={A ConvNet for the 2020s},
      author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
      journal={CVPR},
      year={2022}
    }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
