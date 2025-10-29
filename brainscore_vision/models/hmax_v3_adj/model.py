from brainscore_vision.model_helpers.check_submission import check_models
import functools
import requests
import torch
import torch.nn as nn
import numpy as np
import yaml
from io import BytesIO
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.models.hmax_v3_adj.local_timm.models.RESMAX import hmax_v3_adj

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
    assert name == 'hmax_v3_adj'
    model = load_model()
    print([name for name, _ in model.named_modules()])
    print([name for name, _ in model.named_modules()][7::7])
    preprocessing = functools.partial(load_preprocess_images, image_size=322)
    wrapper = PytorchWrapper(identifier='hmax_v3_adj', model=model, preprocessing=preprocessing)
    wrapper.image_size = 322
    return wrapper

def get_layers(name):
    assert name == 'hmax_v3_adj'
    return ["model_backbone.c1.resizing_layers.2","model_backbone.c2.scoring_conv.pool","model_backbone.c2b_seq.0","model_backbone.global_pool.pool1","model_backbone.s1.layer1.1","model_backbone.s1.layer1.2.bn2","model_backbone.s2.layer.1","model_backbone.s2b.s2b_seqs.0.0","model_backbone.s2b.s2b_seqs.1.0","model_backbone.s2b.s2b_seqs.1.2.conv1","model_backbone.s2b.s2b_seqs.2.1.conv1","model_backbone.s2b.s2b_seqs.2.3.conv1","model_backbone.s2b.s2b_seqs.3.1.conv1","model_backbone.s2b.s2b_seqs.3.3.conv1","model_backbone.s3.layer.0.bn1","model_backbone.s3.layer.2.conv2"]

def get_bibtex(model_identifier):
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)
