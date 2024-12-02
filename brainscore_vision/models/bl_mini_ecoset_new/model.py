from brainscore_vision.model_helpers.check_submission import check_models
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore import score_model
import torchvision.transforms as transforms
import os
import requests
import urllib


"""
Template module for a base model submission to brain-score
"""

# define your custom model here:
class BL_net_64(nn.Module):
    
    def __init__(self, lateral_connections = True, timesteps = 10, LT_position = 'all',
                 classifier_bias = False, norm_type = 'LN'):
        super(BL_net_64, self).__init__()
        # Ensure that kernel_size is always odd, otherwise TransposeConv2d computation will fail
        layer_bias = True # False makes more sense and is in the GitHub Repo of the paper, but in the text it says True
        if norm_type == 'None':
            layer_bias = True
        lt_flag_prelast = 1
        if LT_position == 'last':
            lt_flag_prelast = 0 # if 'last' then only GAP gets a lateral connection
        # Layer 1
        self.conv1 = BLT_Conv(3, 96, 7, lateral_connections, pool_input = False, bias = layer_bias)
        self.conv1_bn = nn.ModuleList([nn.BatchNorm2d(96, momentum=0.99, eps=1e-3) for i in range(timesteps)])
        self.relu1 = nn.ModuleList([nn.ReLU() for i in range(timesteps)])
        # Layer 2
        self.conv2 = BLT_Conv(96, 128, 5, lateral_connections, pool_input = True, bias = layer_bias)
        self.conv2_bn = nn.ModuleList([nn.BatchNorm2d(128, momentum=0.99, eps=1e-3) for i in range(timesteps)])
        self.relu2 = nn.ModuleList([nn.ReLU() for i in range(timesteps)])
        # Layer 3
        self.conv3 = BLT_Conv(128, 192, 3, lateral_connections, pool_input = True, bias = layer_bias)
        self.conv3_bn = nn.ModuleList([nn.BatchNorm2d(192, momentum=0.99, eps=1e-3) for i in range(timesteps)])
        self.relu3 = nn.ModuleList([nn.ReLU() for i in range(timesteps)])
        # Layer 4
        self.conv4 = BLT_Conv(192, 256, 3, lateral_connections, pool_input = True, bias = layer_bias)
        self.conv4_bn = nn.ModuleList([nn.BatchNorm2d(256, momentum=0.99, eps=1e-3) for i in range(timesteps)])
        self.relu4 = nn.ModuleList([nn.ReLU() for i in range(timesteps)])
        # Layer 5
        self.conv5 = BLT_Conv(256, 512, 3, lateral_connections, pool_input = True, bias = layer_bias)
        self.conv5_bn = nn.ModuleList([nn.BatchNorm2d(512, momentum=0.99, eps=1e-3) for i in range(timesteps)])
        self.relu5 = nn.ModuleList([nn.ReLU() for i in range(timesteps)])
        # Layer 6
        self.conv6 = BLT_Conv(512, 1024, 3, lateral_connections, pool_input = True, bias = layer_bias)
        self.conv6_bn = nn.ModuleList([nn.BatchNorm2d(1024, momentum=0.99, eps=1e-3) for i in range(timesteps)])
        self.relu6 = nn.ModuleList([nn.ReLU() for i in range(timesteps)])
        # Layer 7
        self.conv7 = BLT_Conv(1024, 2048, 1, lateral_connections, pool_input = True, bias = layer_bias)
        self.conv7_bn = nn.ModuleList([nn.BatchNorm2d(2048, momentum=0.99, eps=1e-3) for i in range(timesteps)])
        self.relu7 = nn.ModuleList([nn.ReLU() for i in range(timesteps)])
        # Layer 8/ Readout
        self.global_avg_pool = nn.AvgPool2d(2, 2)
        # 100 is the number of classes, should be 565 for full ecoSet and 1000 to check for params number to match the paper (ImageNet)
        self.readout = nn.Linear(2048, 100, bias=classifier_bias)
        self.softmax = nn.Softmax(dim=1)
        # Only for brainscore, no functionality
        self.concatenation_layers = nn.ModuleList([nn.Identity() for i in range(8)])

        self.timesteps = timesteps
    
    def forward(self, inputs):
        with torch.no_grad():
            activations = [[None for t in range(self.timesteps)] for i in range(7)]
            outputs = [None for l in range(self.timesteps)]
            for t in range(self.timesteps):
                if t == 0: # does not accept None as input
                    activations[0][t] = self.relu1[t](self.conv1_bn[t](self.conv1(inputs)))
                    activations[1][t] = self.relu2[t](self.conv2_bn[t](self.conv2(activations[0][t])))
                    activations[2][t] = self.relu3[t](self.conv3_bn[t](self.conv3(activations[1][t])))
                    activations[3][t] = self.relu4[t](self.conv4_bn[t](self.conv4(activations[2][t])))
                    activations[4][t] = self.relu5[t](self.conv5_bn[t](self.conv5(activations[3][t])))
                    activations[5][t] = self.relu6[t](self.conv6_bn[t](self.conv6(activations[4][t])))
                    activations[6][t] = self.relu7[t](self.conv7_bn[t](self.conv7(activations[5][t])))
                else:
                    activations[0][t] = self.relu1[t](self.conv1_bn[t](self.conv1(inputs,activations[0][t-1])))
                    activations[1][t] = self.relu2[t](self.conv2_bn[t](self.conv2(activations[0][t],activations[1][t-1])))
                    activations[2][t] = self.relu3[t](self.conv3_bn[t](self.conv3(activations[1][t],activations[2][t-1])))
                    activations[3][t] = self.relu4[t](self.conv4_bn[t](self.conv4(activations[2][t],activations[3][t-1])))
                    activations[4][t] = self.relu5[t](self.conv5_bn[t](self.conv5(activations[3][t],activations[4][t-1])))
                    activations[5][t] = self.relu6[t](self.conv6_bn[t](self.conv6(activations[4][t],activations[5][t-1])))
                    activations[6][t] = self.relu7[t](self.conv7_bn[t](self.conv7(activations[5][t],activations[6][t-1])))
                pooled_activation = torch.squeeze(self.global_avg_pool(activations[6][t]))
                outputs[t] = torch.log(torch.clamp(self.softmax(self.readout(pooled_activation)),1e-10,1.0))
                # print(f'Calculated output at timestep {t}')
            # Only for brainscore, no functionality
            for layer in range(7):                
                self.concatenation_layers[layer](torch.cat([activations[layer][0], activations[layer][4], activations[layer][9]], dim=1))
            self.concatenation_layers[7](torch.cat([outputs[0], outputs[4], outputs[9]], dim=1))
            return outputs
    
class BLT_Conv(nn.Module):
    # This Conv class takes the input (which can be due to b, l, and/or t) and outputs b, l, t drives for same/other
    # layers.
    
    def __init__(self, in_chan, out_chan, kernel_size, lateral_connection = True, pool_input = True, bias = False):
        super(BLT_Conv, self).__init__()
        self.bottom_up = nn.Conv2d(in_chan, out_chan, kernel_size, bias = False, padding = 'same')
        self.lateral_connect = lateral_connection
        if lateral_connection:
            self.lateral = nn.Conv2d(out_chan, out_chan, kernel_size, bias = False, padding = 'same')
        self.pool_input = pool_input
        if pool_input:
            self.pool = nn.MaxPool2d(2, 2)

    def forward(self, b_input, l_input = None):
        # print(f"Input conv: {b_input.shape}")
        if self.pool_input:
            b_input = self.pool(b_input)
        b_input = self.bottom_up(b_input)
        if self.lateral_connect:
            if l_input is not None:
                l_input = self.lateral(l_input)
        else:
            l_input = None
            
        if l_input is not None:
            output = b_input + l_input
        else:
            output = b_input
        # print(f"Output conv: {output.shape}")
        return output


# init the model and the preprocessing:
# preprocessing = functools.partial(load_preprocess_images, image_size=224)

# def preprocess_images(image_filepaths):
#     images = load_images(image_filepaths)
#     # transform = torch.nn.Sequential(
#     #     transforms.Resize(size=(256, 256),antialias=True),
#     #     transforms.CenterCrop(size=128),
#     #     transforms.ConvertImageDtype(torch.float),
#     #     # transforms.Normalize(mean = np.mean(images)/127.5 - 1., std = np.std(images)/127.5 - 1.)
#     #     transforms.Normalize(mean = [0., 0., 0.], std = [-0.5, -0.5, -0.5])
#     # )
#     # scripted_transforms = torch.jit.script(transform)
#     # images = scripted_transforms(images)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize(size=(256, 256), antialias=True),
#         transforms.CenterCrop(size=128),
#         transforms.ConvertImageDtype(torch.float),
#         # transforms.Normalize(mean = np.mean(images)/127.5 - 1., std = np.std(images)/127.5 - 1.)
#         transforms.Normalize(mean = [0., 0., 0.], std = [-0.5, -0.5, -0.5])
#     ])
#     images = [transform(image) for image in images]
#     return images

# get an activations model from the Pytorch Wrapper
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

cwd = os.getcwd()
url = "https://github.com/thonor111/BL_weights/blob/75a00acf0eed8af0c562cb620abdca894bb0ea48/BL_weights.pth?raw=true"
weight_file_path = os.path.join(cwd, "weights.pth")
urllib.request.urlretrieve(url, weight_file_path)

path = ''
if os.path.isfile(f"{cwd}/weights.pth"):
    path = f"{cwd}/weights.pth"
elif os.path.isfile(f"{cwd}/models/weights.pth"):
    path = f"{cwd}/models/weights.pth"
else:
    path =  f"{cwd}/models/bl_mini_ecoset_new/weights.pth"
# bl_model.load_state_dict(torch.load(f'BL_mini_ecoset_submission/models/BL_weights.pth', weights_only=True)) 
# bl_model.load_state_dict(torch.load(f'BL_weights.pth', map_location=device)) 
bl_model = BL_net_64(timesteps = 10)
bl_model.load_state_dict(torch.load(path, map_location=device))


# preprocessing_small = functools.partial(load_preprocess_images, image_size=224)
activations_model = PytorchWrapper(identifier='bl_mini_ecoset_new', model=bl_model, preprocessing=preprocessing_small)
# activations_model = PytorchWrapper(identifier='bl_mini_ecoset_new', model=bl_model, preprocessing=preprocess_images)
# Layers for brainscore to use
# brainscore_layers = ['relu1.0', 'relu2.0', 'relu3.0', 'relu4.0', 'relu5.0', 'relu6.0', 'relu7.0']
# brainscore_layers = ['relu1.9', 'relu2.9', 'relu3.9', 'relu4.9', 'relu5.9', 'relu6.9', 'relu7.9', 'softmax']
brainscore_layers = ['concatenation_layers.0', 'concatenation_layers.1', 'concatenation_layers.2', 'concatenation_layers.3', 'concatenation_layers.4', 'concatenation_layers.5', 'concatenation_layers.6', 'concatenation_layers.7']
# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='bl_mini_ecoset_new', activations_model=activations_model,
                        # specify layers to consider
                        layers=brainscore_layers)



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
    assert name == 'bl_mini_ecoset_new'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    # print('model was successfully returned')
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
    assert name == 'bl_mini_ecoset_new'
    # brainscore_layers = ['relu1.0', 'relu2.0', 'relu3.0', 'relu4.0', 'relu5.0', 'relu6.0', 'relu7.0']
    # brainscore_layers = ['relu1.9', 'relu2.9', 'relu3.9', 'relu4.9', 'relu5.9', 'relu6.9', 'relu7.9', 'softmax']
    brainscore_layers = ['concatenation_layers.0', 'concatenation_layers.1', 'concatenation_layers.2', 'concatenation_layers.3', 'concatenation_layers.4', 'concatenation_layers.5', 'concatenation_layers.6']

    # returns the layers you want to consider
    # print('Layers were returned')
    return  brainscore_layers

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