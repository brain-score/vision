from brainscore_vision.model_helpers.check_submission import check_models
import functools
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from io import BytesIO
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images


from brainscore_vision.models.hmax_v3_adj2.local_timm.models.RESMAX import hmax_v3_adj

def load_model(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_url = "https://huggingface.co/cmulliken/hmax_v3_adj/resolve/main/model_best.pth.tar"
    args_url = "https://huggingface.co/cmulliken/hmax_v3_adj/resolve/main/args.yaml"

    # Download checkpoint into memory
    response = requests.get(checkpoint_url, stream=True)
    response.raise_for_status()
    checkpoint_file = BytesIO(response.content)

    summmary_response = requests.get(args_url, stream=True)
    summmary_response.raise_for_status()
    args_file = BytesIO(summmary_response.content)

    
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    args_dict = yaml.safe_load(args_file)

    kwargs = {
        'ip_scale_bands': args_dict['model_kwargs']['ip_scale_bands'],
        'classifier_input_size': args_dict['model_kwargs']['classifier_input_size'],
        'bypass': args_dict['model_kwargs']['bypass'],
    }

    model = hmax_v3_adj(
        pretrained=False,
        **kwargs
    )
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def get_model(name):
    assert name == 'hmax_v3_adj2'
    model = load_model()
    print(f"Wrapped model created: {type(model).__name__}")

    preprocessing = functools.partial(load_preprocess_images, image_size=322)
    wrapper = PytorchWrapper(
        identifier='hmax_v3_adj2',
        model=model,
        preprocessing=preprocessing
    )
    wrapper.image_size = 322
    return wrapper

def get_layers(name):
    assert name == 'hmax_v3_adj2'
    return ['model_backbone.s1', 'model_backbone.c1', 'model_backbone.s2', 'model_backbone.c2', 'model_backbone.s2b', 'model_backbone.c2b_score', 'model_backbone.s3']

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)
