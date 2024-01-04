from model_tools.check_submission import check_models
import numpy as np
import torch
import argparse
from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
import torchvision.models
import os

"""
Template module for a base model submission to brain-score
"""
from srf.structured_conv_layer import Srf_layer_shared
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--num_classes', type=int, default=200)

parser.add_argument('--save', type=str, default='./experiment_'+__file__[:-3])
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args, unknown = parser.parse_known_args()
#------------------------------------------------------------------------------

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, init_k, init_order, init_scale,\
                 transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else Srf_layer_shared
        self._layer = module(inC=dim_in + 1,\
                               outC=dim_out,\
                               init_k=init_k,\
                               init_order=init_order,\
                               init_scale=init_scale,\
                               learn_sigma=True,\
                               use_cuda=torch.cuda.is_available())
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.CELU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 2.0, 2.0, 0.0)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 2.0, 2.0, 0.0)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 2]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

# define your custom model here:
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.initialConv = nn.Conv2d(3,32,3,1)

        self.feature_layer1 = ODEBlock(ODEfunc(32))

        self.DS1_norm = norm(32)
        self.DS1_act = nn.CELU(inplace=True)
        self.DS1_conv = nn.Conv2d(32, 64, 4, 2, 1)

        self.feature_layer2 = ODEBlock(ODEfunc(64))

        self.DS2_norm = norm(64)
        self.DS2_act = nn.CELU(inplace=True)
        self.DS2_conv = nn.Conv2d(64, 128, 4, 2, 1)

        self.feature_layer3 = ODEBlock(ODEfunc(128))

        self.fc_norm = norm(128)
        self.fc_act = nn.CELU(inplace=True)
        self.fc_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_flatten = Flatten()
        self.fc_linear = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.initialConv(x)

        x = self.feature_layer1(x)

        x = self.DS1_norm(x)
        x = self.DS1_act(x)
        x = self.DS1_conv(x)

        x = self.feature_layer2(x)

        x = self.DS2_norm(x)
        x = self.DS2_act(x)
        x = self.DS2_conv(x)

        x = self.feature_layer3(x)

        x = self.fc_norm(x)
        x = self.fc_act(x)
        x = self.fc_pool(x)
        x = self.fc_flatten(x)
        x = self.fc_linear(x)

        return x


# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)

# get an activations model from the Pytorch Wrapper
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.is_available(), device)

model = MyModel()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'tinyImageNet_dcn_ode.pth'), map_location = device)['state_dict'])
model = model.module  #Remove DataParallel Wrapper
model.fc_linear = nn.Linear(128,1000) #Imagenet num_classes
print(model)
activations_model = PytorchWrapper(identifier='dcn_ode', model=model, preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='dcn_ode', activations_model=activations_model,
                        # specify layers to consider
                        layers=['feature_layer1.odefunc.conv2', 'feature_layer2.odefunc.conv2', 'feature_layer3.odefunc.conv2'])


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['dcn_ode']


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
    assert name == 'dcn_ode'

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
    assert name == 'dcn_ode'

    # returns the layers you want to consider
    return ['feature_layer1.odefunc.conv2', 'feature_layer2.odefunc.conv2', 'feature_layer3.odefunc.conv2']

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

