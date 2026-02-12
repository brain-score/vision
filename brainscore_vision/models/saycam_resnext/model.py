from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torchvision.models as models
import torch
import functools
import os
import gdown

def get_model(name):
    # 1. Define Architecture
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = torch.nn.Linear(in_features=2048, out_features=6269, bias=True)
    
    # 2. Download Weights (Bypassing Virus Scan)
    # This is the ID from your link: 1lu7RycOTX566_SJrXPCmVZgMumMSy6Ix
    file_id = '1lu7RycOTX566_SJrXPCmVZgMumMSy6Ix'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'saycam_weights.tar'
    
    if not os.path.exists(output):
        # fuzzy=True is essential for bypassing the virus warning on large files
        gdown.download(url, output, quiet=False, fuzzy=True)

    # 3. Load Weights
    # We load to CPU to ensure compatibility with Brain-Score servers
    checkpoint = torch.load(output, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 4. Wrap for Brain-Score
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='saycam_resnext', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    
    return wrapper

def get_layers(name):
    # Standard ResNet layers for Brain-Score
    return ['layer1', 'layer2', 'layer3', 'layer4']

def get_bibtex(model_identifier):
    return """@article{Orhan2020,
              title={Self-supervised learning through the eyes of a child},
              author={Orhan, A. Emin and Gupta, Vivek V. and Lake, Brenden M.},
              journal={NeurIPS},
              year={2020}
              }"""
