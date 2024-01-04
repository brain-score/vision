## Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from model_tools.check_submission import check_models
import numpy as np
import torch
#from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
#from candidate_models import s3 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import reduce
import math
from torchvision.models import resnet50
import os



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


   
import logging
import os
import sys
### S3 download 
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

_logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = 'brainscore2022'
_DEFAULT_REGION = 'us-east-1'
_NO_SIGNATURE = Config(signature_version=UNSIGNED)


def download_folder(folder_key, target_directory, bucket=_DEFAULT_BUCKET, region=_DEFAULT_REGION):
    if not folder_key.endswith('/'):
        folder_key = folder_key + '/'
    s3 = boto3.resource('s3', region_name=region, config=_NO_SIGNATURE)
    bucket = s3.Bucket(bucket)
    bucket_contents = list(bucket.objects.all())
    files = [obj.key for obj in bucket_contents if obj.key.startswith(folder_key)]
    _logger.debug(f"Found {len(files)} files")
    for file in tqdm(files):
        # get part of file after given folder_key
        filename = file[len(folder_key):]
        if len(filename) > 0:
            target_path = os.path.join(target_directory, filename)
            temp_path = target_path + '.filepart'
            bucket.download_file(file, temp_path)
            os.rename(temp_path, target_path)


def download_file(key, target_path, bucket=_DEFAULT_BUCKET, region=_DEFAULT_REGION):
    s3 = boto3.resource('s3', region_name=region, config=_NO_SIGNATURE)
    obj = s3.Object(bucket, key)
    # show progress. see https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
    with tqdm(total=obj.content_length, unit='B', unit_scale=True, desc=key, file=sys.stdout) as progress_bar:
        def progress_hook(bytes_amount):
            progress_bar.update(bytes_amount)

        obj.download_file(target_path, Callback=progress_hook)




# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)
#dir_path = os.path.dirname(os.path.realpath(""))
download_file("MEALV2_ResNet50_380.pth", "MEALV2_ResNet50_380.pth")

#model.load_state_dict(torch.load(dir_path + "/my_model.pth"))
model_ft = resnet50() #models.resnet50(pretrained=True)
ckpt = torch.load("MEALV2_ResNet50_380.pth", map_location = device)
state_dict = ckpt
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace("module.", "")
    new_state_dict[k] = v
state_dict = new_state_dict

model_ft.load_state_dict(new_state_dict)
#model_ft = skgrcnn55() #models.resnet50(pretrained=True)
#model_ft.load_state_dict(torch.load("checkpoint_params_grcnn55_weight_share.pt"))
all_layers = [layer for layer, _ in model_ft.named_modules()]
all_layers = all_layers[1:]

#model_ft.load_state_dict(torch.load("MEALV2_ResNet50_380.pth"))
#all_layers = [layer for layer, _ in model_ft.named_modules()]
#all_layers = all_layers[1:]

#print(all_layers)
model_ft = model_ft.to(device)


# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='resnet50_380', model= model_ft , preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='resnet50_380', activations_model=activations_model,
                        # specify layers to consider
                        layers=all_layers)


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['resnet50_380']


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'resnet50_380'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 380#224
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """

    # quick check to make sure the model is the correct one:
    assert name == 'resnet50_380'

    # returns the layers you want to consider
    return  all_layers

# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return ''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)