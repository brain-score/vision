import functools
import importlib
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import torch.hub
import ssl
from brainscore_vision.model_helpers.s3 import load_weight_file
from torch.nn import Module
from .helpers.helpers import TemporalPytorchWrapper
from pathlib import Path
from urllib.request import urlretrieve

ssl._create_default_https_context = ssl._create_unverified_context


TIME_MAPPINGS = {
        'V1': (50, 100, 1),
        'V2': (70, 100, 2),
        # 'V2': (20, 50, 2),  # MS: This follows from the movshon anesthesized-monkey recordings, so might not hold up
        'V4': (90, 50, 4),
        'IT': (100, 100, 2),
    }


def get_model(identifier: str):

    class Wrapper(Module):
        def __init__(self, model):
            super(Wrapper, self).__init__()
            self.module = model

    mod = importlib.import_module(f'cornet.cornet_s')
    model_ctr = getattr(mod, 'CORnet_S')
    model = model_ctr()
    model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
    # cornet version: shorten identifier
    identifier_short = identifier[:9]
    # cornet version: shorten identifier
    identifier_short = identifier[:9]
    url = f'https://brainscore-storage.s3.us-east-2.amazonaws.com/brainscore-vision/models/ReAlnet/{identifier_short}_best_model_params.pt'
    fh = urlretrieve(url, f'{identifier_short}_best_model_params.pth')
    load_path = fh[0]
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
