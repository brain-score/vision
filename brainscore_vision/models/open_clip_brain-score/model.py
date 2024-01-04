import itertools
from typing import List

import open_clip
from PIL import Image

from model_tools.activations import PytorchWrapper
from model_tools.check_submission import check_models


def get_model_list() -> List[str]:
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    identifiers = [f"{architecture}_{pretrained}" for architecture, pretrained in [
        ('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN50', 'cc12m'),
        ('RN50-quickgelu', 'openai'), ('RN50-quickgelu', 'yfcc15m'), ('RN50-quickgelu', 'cc12m'),
        ('RN101', 'openai'), ('RN101', 'yfcc15m'),
        ('RN101-quickgelu', 'openai'), ('RN101-quickgelu', 'yfcc15m'),
        ('RN50x4', 'openai'), ('RN50x16', 'openai'), ('RN50x64', 'openai'),
        ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion400m_e31'), ('ViT-B-32', 'laion400m_e32'),
        ('ViT-B-32', 'laion2b_e16'), ('ViT-B-32', 'laion2b_s34b_b79k'),
        ('ViT-B-32-quickgelu', 'openai'), ('ViT-B-32-quickgelu', 'laion400m_e31'),
        ('ViT-B-32-quickgelu', 'laion400m_e32'),
        ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion400m_e31'), ('ViT-B-16', 'laion400m_e32'),
        ('ViT-B-16-plus-240', 'laion400m_e31'), ('ViT-B-16-plus-240', 'laion400m_e32'),
        ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion400m_e31'), ('ViT-L-14', 'laion400m_e32'),
        ('ViT-L-14', 'laion2b_s32b_b82k'), ('ViT-L-14-336', 'openai'),
        ('ViT-H-14', 'laion2b_s32b_b79k'),
        ('ViT-g-14', 'laion2b_s12b_b42k'),
        # excluded: ('roberta-ViT-B-32', 'laion2b_s12b_b32k'), ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
        # ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k')
    ]]
    return identifiers


def get_model(identifier: str) -> PytorchWrapper:
    architecture, pretrained = identifier.split('_', maxsplit=1)
    model, _, preprocess = open_clip.create_model_and_transforms(architecture, pretrained=pretrained)
    path_preprocess = lambda paths: [preprocess(Image.open(path)) for path in paths]
    # Note that we are here not normalizing via `image_features /= image_features.norm(dim=-1, keepdim=True)`
    image_model = model.visual  # hoping this is equivalent to `model.encode_image(images)`
    wrapper = PytorchWrapper(identifier=identifier, model=image_model, preprocessing=path_preprocess)
    return wrapper


def get_layers(identifier: str) -> List[str]:
    architecture, pretrained = identifier.split('_', maxsplit=1)
    mapping = {
        'RN50': _resnet_layers([2, 3, 5, 2]),
        'RN101': _resnet_layers([2, 3, 22, 2]),
        'RN50x4': _resnet_layers([3, 5, 9, 5]),
        'ViT-B-32': _vit_layers(12),
        'ViT-B-16': _vit_layers(12),
        'ViT-L-14': _vit_layers(24),
        'ViT-H-14': _vit_layers(32),
        'ViT-g-14': _vit_layers(40),
    }
    return mapping[architecture]


def _resnet_layers(layer_bottlenecks: List[int]):
    return [f"layer{layer + 1}.{bottleneck}"
            for layer, num_bottlenecks in enumerate(layer_bottlenecks) for bottleneck in range(num_bottlenecks)] \
           + ['avgpool']


def _vit_layers(num_blocks: int):
    return list(itertools.chain(*[[f'transformer.resblocks.{block}.ls1', f'transformer.resblocks.{block}.ls2']
                                  for block in range(num_blocks)]))


def get_bibtex(identifier: str) -> str:
    return """@software{ilharco_gabriel_2021_5143773,
      author       = {Ilharco, Gabriel and
                      Wortsman, Mitchell and
                      Wightman, Ross and
                      Gordon, Cade and
                      Carlini, Nicholas and
                      Taori, Rohan and
                      Dave, Achal and
                      Shankar, Vaishaal and
                      Namkoong, Hongseok and
                      Miller, John and
                      Hajishirzi, Hannaneh and
                      Farhadi, Ali and
                      Schmidt, Ludwig},
      title        = {OpenCLIP},
      month        = jul,
      year         = 2021,
      note         = {If you use this software, please cite it as below.},
      publisher    = {Zenodo},
      version      = {0.1},
      doi          = {10.5281/zenodo.5143773},
      url          = {https://doi.org/10.5281/zenodo.5143773}
    }"""


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
