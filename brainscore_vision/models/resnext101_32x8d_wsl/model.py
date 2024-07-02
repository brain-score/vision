import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import torch.hub
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


def get_model():
    model_identifier = "resnext101_32x8d_wsl"
    model = torch.hub.load('facebookresearch/WSL-Images', model_identifier)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    batch_size = {8: 32, 16: 16, 32: 8, 48: 4}
    wrapper = PytorchWrapper(identifier=model_identifier, model=model, preprocessing=preprocessing,
                             batch_size=batch_size[8])
    wrapper.image_size = 224
    return wrapper


def get_layers():
    return (['conv1'] +
            # note that while relu is used multiple times, by default the last one will overwrite all previous ones
            [f"layer{block + 1}.{unit}.relu"
             for block, block_units in enumerate([3, 4, 23, 3]) for unit in range(block_units)] +
            ['avgpool'])


def get_bibtex(model_identifier):
     """
     A method returning the bibtex reference of the requested model as a string.
     """
     return """@inproceedings{mahajan2018exploring,
               title={Exploring the limits of weakly supervised pretraining},
               author={Mahajan, Dhruv and Girshick, Ross and Ramanathan, Vignesh and He, Kaiming and Paluri, Manohar and Li, Yixuan and Bharambe, Ashwin and Van Der Maaten, Laurens},
               booktitle={Proceedings of the European conference on computer vision (ECCV)},
               pages={181--196},
               year={2018}
             }"""

if __name__ == '__main__':
     check_models.check_base_models(__name__) 