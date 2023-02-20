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
# from . import custom_utils
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR)))

import custom_utils

# from .model_metamers_pytorch.model_analysis_folders.all_model_info import ALL_NETWORKS_AND_LAYERS_IMAGES
print(sys.path)
from model_analysis_folders.all_model_info import ALL_NETWORKS_AND_LAYERS_IMAGES

import requests
import tarfile
import sys
import os

MODEL_BASE_PATH='model_metamers_pytorch/model_analysis_folders/visual_networks/'
def load_model_from_build_file(model_name):
    model_directory = ALL_NETWORKS_AND_LAYERS_IMAGES[model_name]['location']
    build_network_spec = importlib.util.spec_from_file_location("build_network",
            os.path.join(model_directory, 'build_network.py'))
    build_network = importlib.util.module_from_spec(build_network_spec)
    build_network_spec.loader.exec_module(build_network)
    try:
        model, ds, metamer_layers = build_network.main(return_metamer_layers=True,
                                                      )
    except ValueError: # Try to re-download the checkpoints
        download_vision_checkpoints()
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
#     return ['cornet_s', 'vgg_19', 'resnet101']
#     return ['resnet50_linf_4_robust', 'resnet50_linf_8_robust', 'resnet50_random_linf8_perturb']
#     return ['resnet50_random_l2_perturb']
#     return ['resnet50_simclr', 'resnet50_moco_v2']
#     return ['hmax_standard']
#     return ['alexnet_early_checkpoint']
#     return ['alexnet_reduced_aliasing_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed']
#     return ['vonealexnet_gaussian_noise_std4_fixed']
#     return ['resnet50_random_l2_perturb']
#     return ['voneresnet50_fixed_noise', 'gvoneresnet50_fixed_noise']
#     return ['resnet50_byol']
#     return ['alexnet', 'alexnet_l2_3_robust', 'alexnet_random_l2_3_perturb',
#             'alexnet_linf_8_robust', 'alexnet_random_linf8_perturb']
#     return ['resnet50', 'resnet50_l2_3_robust', 'resnet50_linf_4_robust', 'resnet50_linf_8_robust',
#             'resnet50_random_l2_perturb', 'resnet50_random_linf8_perturb']
#     return ['texture_shape_resnet50_trained_on_SIN']
#     return ['CLIP_resnet50']
#     return ['CLIP_ViT-B_32']
    return ['SWSL_resnet50'] 
#     return ['vision_transformer_vit_large_patch16_224']
#     return ['texture_shape_alexnet_trained_on_SIN']


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
    return ''

def download_extract_remove(url, extract_location):
    temp_file_location = os.path.join(extract_location, 'temp.tar')
    print('Downloading %s to %s'%(url, temp_file_location))
    with open(temp_file_location, 'wb') as f:
        r = requests.get(url, stream=True)
        for chunk in r.raw.stream(1024, decode_content=False):
            if chunk:
                f.write(chunk)
                f.flush()
    print('Extracting %s'%temp_file_location)
    tar = tarfile.open(temp_file_location)
    tar.extractall(path=extract_location) # untar file into same directory
    tar.close()

    print('Removing temp file %s'%temp_file_location)
    os.remove(temp_file_location)

def download_vision_checkpoints():
    SCRIPT_DIR = os.path.abspath(__file__)
    VISUAL_CHECKPOINTS_LOCATION = os.path.join(os.path.dirname(SCRIPT_DIR), 'model_metamers_pytorch/model_analysis_folders/visual_networks/pytorch_checkpoints/')
    
    # Download the visual checkpoints (~5.5GB)
    url_visual_checkpoints = 'https://mcdermottlab.mit.edu//jfeather/model_metamers_assets/pytorch_metamers_visual_model_checkpoints.tar'
    download_extract_remove(url_visual_checkpoints, VISUAL_CHECKPOINTS_LOCATION)

if __name__ == '__main__':
    check_models.check_base_models(__name__)
