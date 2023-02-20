import sys
import importlib.util
import os

SCRIPT_DIR = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), 'model_metamers_pytorch'))

import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models
from . import custom_utils

# from .model_metamers_pytorch.model_analysis_folders.all_model_info import ALL_NETWORKS_AND_LAYERS_IMAGES
print(sys.path)
from model_analysis_folders.all_model_info import ALL_NETWORKS_AND_LAYERS_IMAGES

MODEL_BASE_PATH='model_metamers_pytorch/model_analysis_folders/visual_networks/'
def load_model_from_build_file(model_name):
    model_directory = ALL_NETWORKS_AND_LAYERS_IMAGES[model_name]['location']
    build_network_spec = importlib.util.spec_from_file_location("build_network",
            os.path.join(model_directory, 'build_network.py'))
    build_network = importlib.util.module_from_spec(build_network_spec)
    build_network_spec.loader.exec_module(build_network)
    model, ds, metamer_layers = build_network.main(return_metamer_layers=True,
                                                  )
    preprocessing = ds.transform_test

    return model, preprocessing

def custom_load_preprocess_images(image_filepaths, transforms, **kwargs):
    images = load_images(image_filepaths)
    all_images = []
    for i in images:
        all_images.append(transforms(i))
    return all_images

def get_model_list():
    return ['resnet50_byol']

def get_model(name):
    model, transforms = load_model_from_build_file(name)
    preprocessing = functools.partial(custom_load_preprocess_images, transforms=transforms)
    wrapper = custom_utils.PytorchWrapperRobustness(identifier=name, model=model, preprocessing=preprocessing)
    if name=='hmax':
        wrapper.image_size=250
    else:
        wrapper.image_size = 224
    return wrapper
  
#     assert name == 'alexnet'
#     model = torchvision.models.alexnet(pretrained=True)
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
#     wrapper = PytorchWrapper(identifier='alexnet', model=model, preprocessing=preprocessing)
#     wrapper.image_size = 224
#     return wrapper


def get_layers(name):
    layers = ALL_NETWORKS_AND_LAYERS_IMAGES[name]['layers']
    return layers
#     assert name == 'alexnet'
#     return ['features.2', 'features.5', 'features.7', 'features.9', 'features.12',
#             'classifier.2', 'classifier.5']

def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return None

if __name__ == '__main__':
    check_models.check_base_models(__name__)
