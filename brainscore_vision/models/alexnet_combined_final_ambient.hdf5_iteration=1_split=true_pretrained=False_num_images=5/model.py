from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import torch
from torch import nn
device = "cpu"
import pytorch_lightning as pl
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import torchvision.models as models
import gdown

# This is an example implementation for submitting custom model named my_custom_model


def get_bibtex(model_identifier):
    return """xx"""


def get_model_list():
  return ['alexnet_combined_final_ambient.hdf5_iteration\=1_split\=true_pretrained\=False_num_images\=5']
          
def get_model(name):
    assert name == 'alexnet_combined_final_ambient.hdf5_iteration\=1_split\=true_pretrained\=False_num_images\=5'
    # https://huggingface.co/models?sort=downloads&search=cvt
    model = models.alexnet(pretrained=False)  # Initialize AlexNet without pretrained weights
    image_size = 224
    url = "https://drive.google.com/file/d/1PwaEh33w2WVZqm8zcI5hPpnTHfP0ebma/view?usp=share_link"
    output = 'epoch\=263-val_loss\=4.065-validation_accuracy\=0.263.ckpt'
    gdown.download(url, output)

    # Wrap the model in PytorchWrapper directly
    checkpoint = torch.load_from_checkpoint(output)
    model.load_state_dict(checkpoint)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)

    return activations_model


def get_layers(name):
    assert name == ''
    layers = []
    #url = "https://drive.google.com/file/d/1PwaEh33w2WVZqm8zcI5hPpnTHfP0ebma/view?usp=share_link"
    #output = 'epoch\=263-val_loss\=4.065-validation_accuracy\=0.263.ckpt'
    #gdown.download(url, output)

    #model = models.alexnet(pretrained=False)  # Initialize AlexNet without pretrained weights
    #checkpoint = torch.load_from_checkpoint(name)
    #model.load_state_dict(checkpoint)
    #for name, module in model.named_modules():
    #        layers.append(name)
    layers = ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
          'classifier.2', 'classifier.5']

    return layers

if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
