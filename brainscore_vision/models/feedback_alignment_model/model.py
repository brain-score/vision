from model_tools.check_submission import check_models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model

"""
Template module for a base model submission to brain-score
"""
class FA_wrapper(nn.Module):
    def __init__(self, module, layer_type, dim):
        super(FA_wrapper, self).__init__()
        self.module = module
        self.layer_type = layer_type
        self.output_grad = None
        self.x_shape = None

        # FA feedback weights definition
        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
        self.reset_weights()

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape
            x = self.module(x)
            x.register_hook(self.FA_hook_post)
            return x
        else:
            return self.module(x)

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False
    
    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            if (self.layer_type == "fc"):
                grad = self.output_grad.mm(self.fixed_fb_weights)
                return torch.clamp(grad, -0.5, 0.5)
            elif (self.layer_type == "conv"):
                grad = torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad)
                return torch.clamp(grad, -0.5, 0.5)
            else:
                raise NameError("=== ERROR: layer type " + str(self.layer_type) + " is not supported in FA wrapper")
        else:
            raise NameError("testing this bit")
            return grad

    def FA_hook_post(self, grad):
        self.output_grad = grad
        return grad


class LinearFA(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearFA, self).__init__()
        
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)

    def forward(self, x):

        x = self.fc(x)

        return x

class ConvFA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvFA, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

        self.conv = FA_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape)
        

    def forward(self, x):

        x = self.conv(x)

        return x


class feedbackAlignmentModel(torch.nn.Module):
    def __init__(self):
        super(feedbackAlignmentModel, self).__init__()

        # First convolutional layer:
        # - input channels = 3 - as CIFAR10 contains RGB images)
        # - output channels = 6 - increasing channel depth to improve feature detection
        # - kernel size = 5 - standard kernel size
        self.conv1 = ConvFA(3, 6, 5)
        self.relu1 = torch.nn.ReLU() # ReLU activation
        
        # Pooling layer:
        # Identify position and translation invariant features
        # kernel size and stride of 2 - result will be 1/4 of the original number of features
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Second convolutional layer:
        # - input channels = 6 - as the pooling layer does not affect number of channels and this was the output from the first conv layer)
        # - output channels = 12 - again increasing channel depth
        # - kernel size = 5 - standard kernel size
        self.conv2 = ConvFA(6, 12, 5)
        self.relu2 = torch.nn.ReLU() # ReLU activation
        
        # First element comes from the batch size (4) multiplied by the output of each kernel (5*5, as kernel size in
        # previous layer is 5), multiplied by the number of channels output from the previous layer (12) (12*5*5*4 = 1200)
        # (This calculation is 'performed' in the flattening operation view() in the forward method)
        self.fully_connected1 = LinearFA(1200, 120)

        # 10 neurons in output layer to correspond to the 10 categories of object
        self.fully_connected2 = LinearFA(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1) # flatten tensor from 12 channels to 1 for the final, linear layers
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        return x


# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=32)

# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='feedback-alignment-model2', model=feedbackAlignmentModel(), preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='feedback-alignment-model2', activations_model=activations_model,
                        # specify layers to consider
                        layers=['conv1', 'conv2', 'relu1', 'relu2', 'fully_connected1'])


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['feedback-alignment-model2']


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
    assert name == 'feedback-alignment-model2'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 32
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
    assert name == 'feedback-alignment-model2'

    # returns the layers you want to consider
    return  ['conv1', 'conv2', 'relu1', 'relu2', 'fully_connected1']

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