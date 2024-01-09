# -*- coding: utf-8 -*-
# +
from brainscore_vision.model_helpers.check_submission import check_models
import torch 
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import os
from timm.models import create_model
from brainscore_vision.model_helpers.activations.pytorch import load_images
import numpy as np
from pathlib import Path
from brainscore_vision.model_helpers import download_weights

BIBTEX = ""
LAYERS = ['blocks.1.blocks.1.0.norm1','blocks.1.blocks.1.0.mlp.act','blocks.1.blocks.1.4.norm2','blocks.2.revert_projs.1.2']

INPUT_SIZE = 240
download_weights(
    bucket='brainscore-vision', 
    folder_path='models/crossvit_v1',
    filename_version_sha=[('submit_crossvit.pth', 'qkn5a_7Hbf7wztbt.N3OoqGiMEzwkZV1', 'e6802429ba85ff80ebf0a2142f16bd12b3db887e')],
    save_directory=Path(__file__).parent)
WEIGHT_PATH = os.path.join(os.path.dirname(__file__),'submit_crossvit.pth')
print(os.path.join(os.path.dirname(__file__)))
# Description of Layers:
# Behavior : 'blocks.2.revert_projs.1.2'
# IT       : 'blocks.1.blocks.1.4.norm2'
# V1       : 'blocks.1.blocks.1.0.norm1'
# V2       : 'blocks.1.blocks.1.0.mlp.act'
# V4       : 'blocks.1.blocks.1.0.mlp.act'

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


def get_model():    
    # Generate Model
    model = create_model('crossvit_15_dagger_240',pretrained = False)
    checkpoint = torch.load(WEIGHT_PATH,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'],strict = True)
    model.eval()

    '''
    pretrained_state_dict = model.state_dict()
    pretrained_state_dict.keys()

    checkpoint = torch.load(WEIGHT_PATH)
    checkpoint2 = torch.load(WEIGHT_PATH)
    for key in checkpoint['state_dict'].keys():
        if key.startswith('module.head.1'):
            print(key,' not modify pretrained')
            checkpoint2['state_dict'][key[7:]] = pretrained_state_dict[key[7:]]
        checkpoint2['state_dict'][key[7:]] = checkpoint['state_dict'][key]
        del checkpoint2['state_dict'][key]
    '''

    ### Load Model and create necessary methods
    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_custom_model, image_size=224)
    # get an activations model from the Pytorch Wrapper  
    wrapper = PytorchWrapper(identifier='crossvit-v1', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
