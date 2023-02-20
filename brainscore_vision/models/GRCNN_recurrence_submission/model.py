import os
import sys
mydir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(mydir)

from GRCNN import GRCNN

from model_tools.check_submission import check_models
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import functools
from model_tools.activations.pytorch import PytorchWrapper
#from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from candidate_models.model_commitments.cornets import CORnetCommitment



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


preprocessing = functools.partial(load_preprocess_images, image_size=224)

#dir_path = os.path.dirname(os.path.realpath(""))
#download_file("checkpoint_params_grcnn55.pt", "checkpoint_params_grcnn55.pt")
#model.load_state_dict(torch.load(dir_path + "/my_model.pth"))
#model_ft.load_state_dict(torch.load("checkpoint_params_grcnn55.pt"))

model_ft = GRCNN([3, 3, 4, 3], [64, 128, 256, 512], SKconv=False, expansion=4, num_classes=1000)
all_layers = [layer for layer, _ in model_ft.named_modules()]
all_layers = all_layers[1:]
all_layers = ['conv1', 'conv2', 'layer1', 'layer1.conv_g_f' , 'layer1.iter_1.8', 
              'layer1.iter_g_1.1', 'layer1.iter_2.8', 'layer1.iter_g_2.1', 'layer1.iter_3.3', 'layer1.iter_3.8' , 
              'layer1.iter_g_3.1', 'layer1.d_conv_1', 'layer1.d_conv_3', 'layer2.conv_g_r', 
              'layer2.iter_1.8', 'layer2.iter_g_1.1', 'layer2.iter_2.8' ,'layer2.iter_g_2.1', 'layer2.iter_3.8', 
              'layer2.iter_g_3.1', 'layer3.conv_f', 'layer3.conv_g_r', 'layer3.iter_1.8', 'layer3.iter_g_1.1' ,
              'layer3.iter_2.8', 'layer3.iter_g_2.1', 'layer3.iter_3.8', 'layer3.iter_g_3.1', 'layer3.iter_4.8', 
              'layer3.iter_g_4.1', 'layer3.d_conv_1e', 'layer4.conv_g_r', 'layer4.iter_1.8', 'layer4.iter_g_1.1', 
              'layer4.iter_2.8', 'layer4.iter_g_2.1', 'layer4.iter_3.8', 'layer4.iter_g_3.1', 
              'lastact.1',  'classifier']
#print(all_layers)
model_ft = model_ft.to(device)


# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='grcnn', model= model_ft , preprocessing=preprocessing)

# # actually make the model, with the layers you want to see specified:
# model = ModelCommitment(identifier='gcrnn', activations_model=activations_model,
#                         # specify layers to consider
#                         layers=all_layers)

#TODO: da scegliere
time_mapping = {'IT': {i : (time_bin_start, time_bin_start + 10) for i,time_bin_start in enumerate(range(70, 250, 10))}}

model = CORnetCommitment(identifier='gcrnn', activations_model=activations_model,
                        # specify layers to consider
                        layers=all_layers,
                        time_mapping=time_mapping)


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['grcnn']


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
    assert name == 'grcnn'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
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
    assert name == 'grcnn'

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
