import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models
# import importlib
# utils = importlib.import_module('sample-model-submission.utils')
# get_clip_vision_model = utils.get_clip_vision_model
from utils import get_clip_vision_model


def get_model_list():
    return ['RN50', 'RN50x4', 'RN50x16', 'RN101']


def get_model(name):
    if name in ['RN50', 'RN50x4', 'RN50x16', 'RN101']:
        # model = get_clip_vision_model(name)
        from torch import nn
        model = nn.Sequential(get_clip_vision_model(name), nn.Linear(1024,1000))
        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        wrapper = PytorchWrapper(identifier=f'GazivNet_{name}', model=model, preprocessing=preprocessing)
        wrapper.image_size = 224
        return wrapper
    else:
        return NotImplementedError


def get_layers(name):
    assert name in ['RN50', 'RN50x4', 'RN50x16', 'RN101']
    # return ['conv1', 'bn1', 'relu', 'conv2', 'bn2', 'relu', 'conv3', 'bn3', 'relu', 'avgpool'] + [f'layer{i}' for i in [1, 2, 3, 4]]
    layers = ['relu1', 'relu2', 'relu3', 'avgpool'] + [f'layer{i}' for i in [1, 2, 3, 4]]
    return [f'0.{layer}' for layer in layers]


def get_bibtex(model_identifier):
    return """@misc{radford2021learning,
                title={Learning Transferable Visual Models From Natural Language Supervision}, 
                author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
                year={2021},
                eprint={2103.00020},
                archivePrefix={arXiv},
                primaryClass={cs.CV}
                }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
