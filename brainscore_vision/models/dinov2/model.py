import functools
import os
from pathlib import Path
import torch
from transformers import AutoImageProcessor, Dinov2ForImageClassification

# import sys
# sys.path.append('../../../')
from brainscore_vision.model_helpers.check_submission import check_models

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images




def get_model_list():
    return ['dinov2']


def get_model(name):
    assert name == 'dinov2'
    model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='dinov2', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


# def get_layers(name):
#     assert name == 'dinov2'
#     return ['dinov2.encoder.layer.5']

LAYERS = ['dinov2.encoder.layer.5']


def get_bibtex(model_identifier):
    return """@misc{oquab2023dinov2,
                    title={DINOv2: Learning Robust Visual Features without Supervision},
                    author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
                    journal={arXiv:2304.07193},
                    year={2023}
                    }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
