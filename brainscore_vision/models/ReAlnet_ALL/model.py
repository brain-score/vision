import functools
import importlib
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import torch.hub
import ssl
from brainscore_core.supported_data_standards.brainio.s3 import load_weight_file
from torch.nn import Module
from .helpers.helpers import TemporalPytorchWrapper
from pathlib import Path
from brainscore_core.supported_data_standards.brainio.s3 import load_file
import json
import os

ssl._create_default_https_context = ssl._create_unverified_context


TIME_MAPPINGS = {
        'V1': (50, 100, 1),
        'V2': (70, 100, 2),
        # 'V2': (20, 50, 2),  # MS: This follows from the movshon anesthesized-monkey recordings, so might not hold up
        'V4': (90, 50, 4),
        'IT': (100, 100, 2),
    }


def load_config(json_file):
    # Get the directory containing this script (model.py)
    base_dir = os.path.dirname(__file__)
    
    # Construct the path to the JSON file
    json_path = os.path.join(base_dir, json_file)
    
    # Read the JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_model(identifier: str):

    class Wrapper(Module):
        def __init__(self, model):
            super(Wrapper, self).__init__()
            self.module = model

    mod = importlib.import_module(f'cornet.cornet_s')
    model_ctr = getattr(mod, 'CORnet_S')
    model = model_ctr()
    model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix

    # only get the numerical number from the identifier
    identifier_number = ''.join(filter(str.isdigit, identifier))
    # load the weights from the json file
    weights_info = load_config("weights.json")
    version_id = weights_info['version_ids'][identifier]
    # load the weights from the s3 bucket

    load_path = load_file(bucket="brainscore-storage", folder_name="brainscore-vision/models/user_488/",
                      relative_path=f"sub-{identifier_number}.pt",
                      version_id=version_id,
                      )

    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)  # map onto cpu
    new_state_dict = {}
    for key, val in checkpoint.items():
        # remove "module." (if it exists) from the key
        new_key = key.replace("realnet.", "")
        # discard the keys starting with "fc" 
        if not new_key.startswith('fc'):
            new_state_dict[new_key] = val

    model.load_state_dict(new_state_dict)
    model = model.module  # unwrap
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing,
                                     separate_time=True)
    wrapper.image_size = 224
    return wrapper




def get_layers(identifier: str):
    return (['V1.output-t0'] +
               [f'{area}.output-t{timestep}'
                for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                for timestep in timesteps] +
               ['decoder.avgpool-t0']
            )
