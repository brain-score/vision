import functools

import torchvision.models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

BIBTEX = """@inproceedings{szegedy2016rethinking,
  title={Rethinking the inception architecture for computer vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2818--2826},
  year={2016}
}"""

# these layer choices were not investigated in any depth, we blindly picked all high-level blocks
LAYERS = [
    "Conv2d_1a_3x3",
    "maxpool1",
    "Conv2d_3b_1x1",
    "maxpool2",
    "Mixed_5b",
    "Mixed_5c",
    "Mixed_5d",
    "Mixed_6a",
    "Mixed_6b",
    "Mixed_6c",
    "Mixed_6d",
    "Mixed_6e",
    "Mixed_7a",
    "Mixed_7b",
    "Mixed_7c",
    "avgpool",
]

def get_model():
    model = torchvision.models.inception_v3(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(
        identifier="inception_v3", model=model, preprocessing=preprocessing
    )
    wrapper.image_size = 224
    return wrapper
