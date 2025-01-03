from torch import nn
import torch
from torch.nn import functional as F
from typing import Optional, Dict

from code_.model_activations.models.layer_operations.preset_filters import filters  

torch.manual_seed(42)

INPUT_CHANNELS = 3

class WaveletConvolution(nn.Module):
    """
    A convolutional layer that uses pre-defined filters for processing images.

    Attributes:
        filter_type (str): The type of filter to use (e.g., 'curvature', 'gabor').
        filter_size (int): The size of the filters.
        filter_params (Dict, optional): Parameters specific to the type of filters to use.
        device (str, optional): The device on which to perform calculations.
    """
    def __init__(self, filter_type: str, filter_params: Optional[Dict] = None, filter_size: int = 15, device: Optional[str] = None) -> None:
        super().__init__()
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.filter_params = get_kernel_params(self.filter_type)
        self.layer_size = get_layer_size(self.filter_type, self.filter_params)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through the conv layer.

        Args:
            x (torch.Tensor): The input tensor to be convolved.

        Returns:
            torch.Tensor: The convolved output tensor.
        """
        x = x.to(self.device)
        in_channels = x.shape[1]
        convolved_tensor = []

        weights = filters(in_channels=1, kernel_size=self.filter_size, filter_type=self.filter_type, filter_params=self.filter_params).to(self.device)

        for i in range(in_channels):
            channel_image = x[:, i:i+1, :, :]
            channel_convolved = F.conv2d(channel_image, weight=weights, padding=weights.shape[-1] // 2 - 1)
            convolved_tensor.append(channel_convolved)

        return torch.cat(convolved_tensor, dim=1)


def initialize_conv_layer(conv_layer: nn.Conv2d, initialization: str) -> None:
    """
    Initializes the weights of a convolutional layer based on the specified method.

    Args:
        conv_layer (nn.Conv2d): The convolutional layer to be initialized.
        initialization (str): The method to use for initialization based on torch's init methods ('kaiming_uniform', 'kaiming_normal', etc.).

    Raises:
        AssertionError: If the initialization method is not supported.
    """
    init_types = ['kaiming_uniform', 'kaiming_normal', 'orthogonal', 'xavier_uniform', 'xavier_normal', 'uniform', 'normal']
    assert initialization in init_types, f'Invalid initialization type, choose one of {init_types}'

    match initialization:
        case 'kaiming_uniform':
            nn.init.kaiming_uniform_(conv_layer.weight)
        case 'kaiming_normal':
            nn.init.kaiming_normal_(conv_layer.weight)
        case 'orthogonal':
            nn.init.orthogonal_(conv_layer.weight)
        case 'xavier_uniform':
            nn.init.xavier_uniform_(conv_layer.weight)
        case 'xavier_normal':
            nn.init.xavier_normal_(conv_layer.weight)
        case 'uniform':
            nn.init.uniform_(conv_layer.weight, a=0, b=1)
        case 'normal':
            nn.init.normal_(conv_layer.weight, mean=0, std=1)


def get_kernel_params(kernel_type: str) -> Dict:
    """
    Retrieves kernel parameters based on the kernel type.

    Args:
        kernel_type (str): The type of kernel for which to get parameters.

    Returns:
        Dict: A dictionary containing parameters for the specified kernel type.

    Raises:
        ValueError: If the kernel type is not supported.
    """
    if kernel_type == 'curvature':
        return {'n_ories': 12, 'n_curves': 3, 'gau_sizes': (5,), 'spatial_fre': [1.2]}
    elif kernel_type == 'gabor':
        return {'n_ories': 12, 'num_scales': 3}
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")



def get_layer_size(kernel_type: str, kernel_params: Dict) -> int:
    """
    Computes the number of output channels based on the kernel type and its parameters.

    Args:
        kernel_type (str): The kernel type.
        kernel_params (Dict): Parameters for the kernel.

    Returns:
        int: Number of output channels.

    Raises:
        ValueError: If the kernel type is not supported.
    """
    if kernel_type == 'curvature':
        return kernel_params['n_ories'] * kernel_params['n_curves'] * len(kernel_params['gau_sizes']) * len(kernel_params['spatial_fre']) * INPUT_CHANNELS
    elif kernel_type == 'gabor':
        return kernel_params['n_ories'] * kernel_params['num_scales'] * INPUT_CHANNELS
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")