# -*- coding: utf-8 -*-
# +
from model_tools.check_submission import check_models
import torch 
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
import os
from timm.models import create_model
from model_tools.activations.pytorch import load_images
import numpy as np
import wget

INPUT_SIZE = 256
CROP_SIZE  = 224
BATCH_SIZE = 64
MODEL_IDENTIFIER = f'ViT-Base-Patch-16-224-PRETRAINED-INPUT-SIZE-{INPUT_SIZE}-CROP-SIZE-{CROP_SIZE}-V1'
LAYERS = ['blocks.1.mlp.act']

PATH_SUB = os.path.dirname(__file__)
print(os.path.join(os.path.dirname(__file__)))
# Description of Layers:
# Behavior : pre_logits
# IT       : ['blocks.8.norm1']
# V1       : ['blocks.1.mlp.act']
# V2       : ['blocks.6.norm2']
# V4       : ['blocks.2.mlp.act']

# +
def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.CenterCrop((image_size,image_size)),
        torchvision_preprocess(**kwargs),
    ])


def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])

def load_preprocess_custom_model(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


# -

# Generate Model
model = create_model('vit_base_patch16_224',pretrained = True)
model.eval()

### Load Model and create necessary methods
# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_custom_model, image_size=CROP_SIZE)
# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier=MODEL_IDENTIFIER, model=model, preprocessing=preprocessing,batch_size = BATCH_SIZE)
# actually make the model, with the layers you want to see specified:
#model = ModelCommitment(identifier='custom_model_v1', activations_model=activations_model,layers=LAYERS)

def get_model_list():
    return [MODEL_IDENTIFIER]


def get_model(name):
    assert name == MODEL_IDENTIFIER   
    wrapper = activations_model
    wrapper.image_size = CROP_SIZE
    return wrapper


def get_layers(name):
    assert name == MODEL_IDENTIFIER
    return LAYERS


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
