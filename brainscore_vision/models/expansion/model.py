from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import math
from typing import Optional, Tuple
from .layer_operations.convolution import WaveletConvolution, initialize_conv_layer
from .layer_operations.output import Output
from .layer_operations.nonlinearity import NonLinearity

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class Model(torch.nn.Module):
    """Expansion model architecture consisting of 5 convolutional and pooling layers."""
    def __init__(self, conv1: torch.nn.Module, pool1: torch.nn.Module, conv2: torch.nn.Module, pool2: torch.nn.Module, 
                 conv3: torch.nn.Module, pool3: torch.nn.Module, conv4: torch.nn.Module, pool4: torch.nn.Module, 
                 conv5: torch.nn.Module, pool5: torch.nn.Module, nl: torch.nn.Module, last: torch.nn.Module, device: str) -> None:
        super(Model, self).__init__()
        self.conv1 = conv1
        self.pool1 = pool1
        self.conv2 = conv2
        self.pool2 = pool2
        self.conv3 = conv3
        self.pool3 = pool3
        self.conv4 = conv4
        self.pool4 = pool4
        self.conv5 = conv5
        self.pool5 = pool5
        self.nl = nl
        self.last = last
        self.device = device
        
    def forward(self, x)-> torch.Tensor:
        """Forward pass through the network, applying each layer sequentially."""
        x = x.to(self.device)
        # Processing layers 1 to 5
        for i in range(1, 6):
            conv = getattr(self, f'conv{i}')
            pool = getattr(self, f'pool{i}')
            x = conv(x)
            x = self.nl(x)
            x = pool(x)
        # Output layer
        x = self.last(x)
        return x

class Expansion5L:
    """Constructing the 5-layer expansion model with customizable filter sizes and types."""
    def __init__(self, filters_1: Optional[int] = None, filters_2: int = 1000, filters_3: int = 3000, 
                 filters_4: int = 5000, filters_5: int = 3000, init_type: str = 'kaiming_uniform', 
                 non_linearity: str = 'relu', device: str = 'cuda') -> None:
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.filters_4 = filters_4
        self.filters_5 = filters_5
        self.init_type = init_type
        self.non_linearity = non_linearity
        self.device = device

    def create_layer(self, in_filters: int, out_filters: int, kernel_size: Tuple[int, int], 
                     stride: int = 1, pool_kernel: int = 2, pool_stride: Optional[int] = None, 
                     padding: int = 0) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Creates a convolutional layer and a pooling layer with either fixed or random conv filters"""
        conv = torch.nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=False).to(self.device)
        initialize_conv_layer(conv, self.init_type)
        pool = torch.nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        return conv, pool

    def build(self):
        """Builds the complete model using specified configurations."""
        # Pre-fixed or random filters for layer 1
        if self.filters_1 is None:
            conv1 = WaveletConvolution(filter_size=15, filter_type='curvature', device=self.device)
            pool1 = torch.nn.AvgPool2d(kernel_size=2)
            self.filters_1 = conv1.layer_size
        else:
            padding = math.floor(15 / 2)
            conv1, pool1 = self.create_layer(3, self.filters_1, (15, 15), padding=padding)

        # Setup layers 2 to 5
        conv2, pool2 = self.create_layer(self.filters_1, self.filters_2, (7, 7), 1, 2)
        conv3, pool3 = self.create_layer(self.filters_2, self.filters_3, (5, 5), 1, 2)
        conv4, pool4 = self.create_layer(self.filters_3, self.filters_4, (3, 3), 1, 2)
        conv5, pool5 = self.create_layer(self.filters_4, self.filters_5, (3, 3), 1, 4, 1)

        nl = NonLinearity(self.non_linearity)
        last = Output()

        return Model(
            conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5, pool5, nl, last, self.device
        )

def get_model_list():
    return ['expansion']

def get_model(name):
    assert name == 'expansion'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='expansion', model=Expansion5L().build(), preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'expansion'
    return ['last']


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)

